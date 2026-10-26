//! Records how many slots each validator is scheduled to lead in an epoch into their
//! respective ValidatorHistory accounts.
//!
//! The steward measures alpenglow leader rewards against these counts, so a validator's
//! participation ratio doesn't depend on how many leader slots it happened to draw.
//! The leader schedule is fixed before the epoch starts, so one `getLeaderSchedule`
//! call covers every validator, unlike the block metadata operation, which walks
//! every block.
//!
//! Leader slots belong to vote accounts (SIMD-0180), and alpenglow pays the leader half of a
//! slot's reward to that vote account. `getLeaderSchedule` keys the schedule by node identity,
//! which is mapped back to vote accounts here. RPC v4.4 adds `keyByVoteAccount`, which can
//! replace the mapping once RPC nodes run it.
//!
//! The operation runs once at 10% epoch completion, after the vote account operation
//! has created the epoch's entries. It also fills in any of the previous
//! [`LEADER_SCHEDULE_LOOKBACK_EPOCHS`] epochs that are missing a count, so a keeper outage
//! doesn't leave gaps while RPC still serves those schedules.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use solana_client::{nonblocking::rpc_client::RpcClient, rpc_response::RpcVoteAccountInfo};
use solana_metrics::{datapoint_error, datapoint_info};
use solana_sdk::{
    epoch_info::EpochInfo,
    pubkey::Pubkey,
    signature::{Keypair, Signer},
};
use stakenet_sdk::{
    models::{entries::UpdateInstruction, errors::JitoTransactionError, submit_stats::SubmitStats},
    utils::{helpers::is_live_vote_account, transactions::submit_instructions},
};
use validator_history::ValidatorHistory;

use crate::{
    entries::leader_slots_entry::LeaderSlotsEntry,
    state::{keeper_config::KeeperConfig, keeper_state::KeeperState},
};

use super::keeper_operations::{check_flag, KeeperOperations};

/// Number of past epochs checked for missing counts. RPC nodes keep stakes for the current
/// epoch and the four before it, so older schedules usually can't be fetched.
pub const LEADER_SCHEDULE_LOOKBACK_EPOCHS: u64 = 4;

/// Manages recording scheduled leader slot counts into validator history accounts.
pub struct UpdateLeaderSlotsOperation<'a> {
    /// RPC Client
    client: Arc<RpcClient>,

    /// Oracle authority keypair
    oracle_authority_keypair: Arc<Keypair>,

    /// Validator History Program ID
    program_id: Pubkey,

    /// Keeper Config
    keeper_config: &'a KeeperConfig,

    /// Keeper State
    keeper_state: &'a KeeperState,
}

impl<'a> UpdateLeaderSlotsOperation<'a> {
    /// Creates a new operation from the keeper's config and current state.
    pub fn new(keeper_config: &'a KeeperConfig, keeper_state: &'a KeeperState) -> Self {
        Self {
            client: keeper_config.client.clone(),
            oracle_authority_keypair: keeper_config
                .oracle_authority_keypair
                .clone()
                .expect("UpdateLeaderSlotsOperation requires oracle_authority_keypair"),
            program_id: keeper_config.validator_history_program_id,
            keeper_config,
            keeper_state,
        }
    }

    /// Returns the [`KeeperOperations`] variant for this operation.
    fn operation() -> KeeperOperations {
        KeeperOperations::UpdateLeaderSlots
    }

    /// Returns `true` when the operation should execute.
    ///
    /// Runs once per epoch after 10% slot completion.
    fn should_run(epoch_info: &EpochInfo, runs_for_epoch: u64) -> bool {
        epoch_info.slot_index > epoch_info.slots_in_epoch * 10 / 100 && runs_for_epoch < 1
    }

    /// Entry point for the operation. Checks whether the operation should run,
    /// executes it, and returns updated run/error/transaction counts for the epoch.
    pub async fn fire(&self) -> (KeeperOperations, u64, u64, u64) {
        let operation = Self::operation();

        let (mut runs_for_epoch, mut errors_for_epoch, mut txs_for_epoch) = self
            .keeper_state
            .copy_runs_errors_and_txs_for_epoch(operation);

        let should_run = Self::should_run(&self.keeper_state.epoch_info, runs_for_epoch)
            && check_flag(self.keeper_config.run_flags, operation);

        if should_run {
            match self.process().await {
                Ok(stats) => {
                    for message in stats.results.iter() {
                        if let Err(e) = message {
                            datapoint_error!(
                                "update-leader-slots-error",
                                ("error", e.to_string(), String),
                            );
                            errors_for_epoch += 1;
                        } else {
                            txs_for_epoch += 1;
                        }
                    }
                    if stats.errors == 0 {
                        runs_for_epoch += 1;
                    }
                }
                Err(e) => {
                    datapoint_error!(
                        "update-leader-slots-error",
                        ("error", e.to_string(), String),
                    );
                    errors_for_epoch += 1;
                }
            }
        }

        (operation, runs_for_epoch, errors_for_epoch, txs_for_epoch)
    }

    /// Fetches the leader schedule of the current epoch and of each lookback epoch RPC still
    /// serves, counts each validator's slots, and submits `UpdateLeaderSlots` instructions for
    /// every live validator history account whose count can be attributed and isn't recorded yet.
    async fn process(&self) -> Result<SubmitStats, JitoTransactionError> {
        let current_epoch = self.keeper_state.epoch_info.epoch;
        let epoch_schedule = &self.keeper_state.epoch_schedule;

        // Filter out closed/reassigned vote accounts
        let live_vote_accounts: Vec<Pubkey> = self
            .keeper_state
            .all_history_vote_account_map
            .iter()
            .filter(|(_, account)| is_live_vote_account(account.as_ref()))
            .map(|(vote_account, _)| *vote_account)
            .collect();

        let mut update_instructions = Vec::new();
        let first_epoch = current_epoch.saturating_sub(LEADER_SCHEDULE_LOOKBACK_EPOCHS);
        for epoch in (first_epoch..=current_epoch).rev() {
            let leader_schedule = self
                .client
                .get_leader_schedule(Some(epoch_schedule.get_first_slot_in_epoch(epoch)))
                .await
                .map_err(|e| JitoTransactionError::Custom(e.to_string()))?;
            let Some(leader_schedule) = leader_schedule else {
                if epoch == current_epoch {
                    return Err(JitoTransactionError::Custom(format!(
                        "Leader schedule unavailable epoch={epoch}"
                    )));
                }
                break;
            };

            let leader_slots_by_vote = leader_slots_by_vote_account(
                &leader_schedule,
                &self.keeper_state.vote_account_map,
                &live_vote_accounts,
                epoch_schedule.get_slots_in_epoch(epoch),
            )
            .map_err(JitoTransactionError::Custom)?;

            let epoch_instructions: Vec<_> = leader_slots_by_vote
                .iter()
                .filter(|(vote_account, &total_leader_slots)| {
                    needs_leader_slots_update(
                        self.keeper_state.validator_history_map.get(vote_account),
                        epoch,
                        current_epoch,
                        total_leader_slots,
                    )
                })
                .map(|(vote_account, &total_leader_slots)| {
                    LeaderSlotsEntry::new(
                        *vote_account,
                        &self.program_id,
                        &self.oracle_authority_keypair.pubkey(),
                        epoch,
                        total_leader_slots,
                    )
                    .update_instruction()
                })
                .collect();

            datapoint_info!(
                "update-leader-slots",
                ("epoch", epoch, i64),
                ("live-vote-accounts", live_vote_accounts.len(), i64),
                ("attributed-vote-accounts", leader_slots_by_vote.len(), i64),
                ("updates", epoch_instructions.len(), i64),
            );

            update_instructions.extend(epoch_instructions);
        }

        submit_instructions(
            &self.client,
            update_instructions,
            &self.oracle_authority_keypair,
            self.keeper_config.priority_fee_in_microlamports,
            self.keeper_config.tx_retry_count,
            self.keeper_config.tx_confirmation_seconds,
            None,
            self.keeper_config.no_pack,
        )
        .await
        .map_err(|e| e.into())
    }
}

/// Returns `true` when `total_leader_slots` should be written for `epoch`.
///
/// A past epoch is only written when its entry exists, since the program only inserts entries
/// in order. The current epoch's entry is created if it doesn't exist yet.
pub fn needs_leader_slots_update(
    validator_history: Option<&ValidatorHistory>,
    epoch: u64,
    current_epoch: u64,
    total_leader_slots: u32,
) -> bool {
    let entry = validator_history.and_then(|validator_history| {
        validator_history
            .history
            .arr
            .iter()
            .find(|entry| u64::from(entry.epoch) == epoch)
    });
    match entry {
        Some(entry) => entry.total_leader_slots != total_leader_slots,
        None => epoch == current_epoch,
    }
}

/// Counts the slots each of `live_vote_accounts` is scheduled to lead in `leader_schedule`, which
/// is keyed by node identity.
///
/// Identities are mapped with each vote account's current `node_pubkey`, so a known identity
/// without slots counts as zero. The identity view merges vote accounts that share an identity,
/// so those are skipped, as are vote accounts without a known identity.
///
/// Errors, so nothing is written, when the schedule doesn't cover the epoch or is keyed by vote
/// account.
pub fn leader_slots_by_vote_account(
    leader_schedule: &HashMap<String, Vec<usize>>,
    vote_accounts: &HashMap<Pubkey, RpcVoteAccountInfo>,
    live_vote_accounts: &[Pubkey],
    slots_in_epoch: u64,
) -> Result<HashMap<Pubkey, u32>, String> {
    let scheduled_slots: u64 = leader_schedule
        .values()
        .map(|slots| slots.len() as u64)
        .sum();
    if scheduled_slots != slots_in_epoch {
        return Err(format!(
            "Leader schedule has {scheduled_slots} slots, epoch has {slots_in_epoch}"
        ));
    }

    let vote_pubkeys: HashSet<&str> = vote_accounts
        .values()
        .map(|vote_account_info| vote_account_info.vote_pubkey.as_str())
        .collect();
    if leader_schedule
        .keys()
        .any(|key| vote_pubkeys.contains(key.as_str()))
    {
        return Err("Leader schedule is keyed by vote account, expected node identity".to_string());
    }

    let mut vote_accounts_by_identity: HashMap<&str, usize> = HashMap::new();
    for vote_account_info in vote_accounts.values() {
        *vote_accounts_by_identity
            .entry(vote_account_info.node_pubkey.as_str())
            .or_default() += 1;
    }

    Ok(live_vote_accounts
        .iter()
        .filter_map(|vote_account| {
            let identity = vote_accounts.get(vote_account)?.node_pubkey.as_str();
            if vote_accounts_by_identity[identity] > 1 {
                return None;
            }
            let slots = leader_schedule.get(identity).map_or(0, Vec::len);
            Some((*vote_account, u32::try_from(slots).ok()?))
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use bytemuck::Zeroable;
    use validator_history::{CircBuf, ValidatorHistoryEntry};

    use super::*;

    fn vote_account_info(vote_account: &Pubkey, identity: &Pubkey) -> RpcVoteAccountInfo {
        RpcVoteAccountInfo {
            vote_pubkey: vote_account.to_string(),
            node_pubkey: identity.to_string(),
            activated_stake: 0,
            commission: 0,
            epoch_vote_account: true,
            epoch_credits: vec![],
            last_vote: 0,
            root_slot: 0,
        }
    }

    /// Slot indexes for `windows` leader windows of 4 slots, starting at window `first`.
    fn windows(first: usize, windows: usize) -> Vec<usize> {
        (first * 4..(first + windows) * 4).collect()
    }

    struct Cluster {
        vote_accounts: HashMap<Pubkey, RpcVoteAccountInfo>,
        vote: [Pubkey; 4],
        identity: [Pubkey; 3],
    }

    /// Vote accounts 0 and 1 have their own identities; 2 and 3 share identity 2.
    fn cluster() -> Cluster {
        let vote = [(); 4].map(|_| Pubkey::new_unique());
        let identity = [(); 3].map(|_| Pubkey::new_unique());
        let vote_accounts = [(0, 0), (1, 1), (2, 2), (3, 2)]
            .into_iter()
            .map(|(v, i)| (vote[v], vote_account_info(&vote[v], &identity[i])))
            .collect();
        Cluster {
            vote_accounts,
            vote,
            identity,
        }
    }

    #[test]
    fn test_leader_slots_by_vote_account() {
        let Cluster {
            vote_accounts,
            vote,
            identity,
        } = cluster();
        let leader_schedule = HashMap::from([
            (identity[0].to_string(), windows(0, 2)),
            (identity[2].to_string(), windows(2, 3)),
            (Pubkey::new_unique().to_string(), windows(5, 1)),
        ]);
        let unknown = Pubkey::new_unique();

        let counts = leader_slots_by_vote_account(
            &leader_schedule,
            &vote_accounts,
            &[vote[0], vote[1], vote[2], vote[3], unknown],
            24,
        )
        .unwrap();
        assert_eq!(counts[&vote[0]], 8);
        // Known identity without slots this epoch is a real zero
        assert_eq!(counts[&vote[1]], 0);
        // A shared identity's slots can't be split between its vote accounts, and a vote
        // account without a known identity can't be looked up
        assert_eq!(counts.len(), 2);
    }

    #[test]
    fn test_leader_slots_by_vote_account_rejects_unusable_schedules() {
        let Cluster {
            vote_accounts,
            vote,
            identity,
        } = cluster();

        // Keyed by vote account
        let by_vote_account = HashMap::from([
            (vote[0].to_string(), windows(0, 2)),
            (identity[1].to_string(), windows(2, 4)),
        ]);
        assert!(leader_slots_by_vote_account(&by_vote_account, &vote_accounts, &vote, 24).is_err());

        // Doesn't cover the epoch
        let partial = HashMap::from([(identity[0].to_string(), windows(0, 2))]);
        assert!(leader_slots_by_vote_account(&partial, &vote_accounts, &vote, 24).is_err());
    }

    /// A validator history with entries for `epochs`, each recording its leader slot count.
    fn validator_history(epochs: &[(u16, u32)]) -> Box<ValidatorHistory> {
        let mut validator_history = Box::new(ValidatorHistory::zeroed());
        validator_history.history = CircBuf::default();
        for &(epoch, total_leader_slots) in epochs {
            validator_history.history.push(ValidatorHistoryEntry {
                epoch,
                total_leader_slots,
                ..ValidatorHistoryEntry::default()
            });
        }
        validator_history
    }

    #[test]
    fn test_needs_leader_slots_update() {
        let unset = ValidatorHistoryEntry::default().total_leader_slots;
        let history = validator_history(&[(900, unset), (901, 216), (902, unset)]);

        // Missing or different counts are written, recorded ones are skipped
        assert!(needs_leader_slots_update(Some(&history), 900, 902, 216));
        assert!(needs_leader_slots_update(Some(&history), 901, 902, 200));
        assert!(!needs_leader_slots_update(Some(&history), 901, 902, 216));

        // The current epoch's entry is created, a past epoch's isn't
        assert!(needs_leader_slots_update(Some(&history), 903, 903, 216));
        assert!(needs_leader_slots_update(None, 903, 903, 216));
        assert!(!needs_leader_slots_update(Some(&history), 899, 903, 216));
        assert!(!needs_leader_slots_update(None, 899, 903, 216));
    }
}
