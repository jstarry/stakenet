use std::{
    collections::{HashMap, HashSet},
    fs::File,
    path::PathBuf,
    str::FromStr,
    sync::Arc,
};

use anyhow::anyhow;
use clap::Parser;
use futures_util::future::join_all;
use solana_client::nonblocking::rpc_client::RpcClient;
use solana_sdk::{
    pubkey::Pubkey, signature::read_keypair_file, signer::Signer, transaction::Transaction,
};
use stakenet_keeper::{
    entries::leader_slots_entry::LeaderSlotsEntry,
    operations::update_leader_slots::{leader_slots_by_vote_account, needs_leader_slots_update},
};
use stakenet_sdk::{
    models::entries::UpdateInstruction,
    utils::{
        accounts::get_all_validator_history_accounts, helpers::is_live_vote_account,
        transactions::get_vote_accounts_with_retry,
    },
};
use validator_history::constants::MIN_VOTE_EPOCHS;

#[derive(Parser)]
#[command(about = "Crank to record scheduled leader slots to validator history accounts")]
pub struct CrankUpdateLeaderSlots {
    /// Path to the oracle authority keypair for transaction signing
    #[arg(short, long, env, default_value = "~/.config/solana/id.json")]
    keypair_path: PathBuf,

    /// Epoch number
    #[arg(long)]
    epoch: Option<u64>,

    /// Path to the `result` of a `getLeaderSchedule` call for the epoch, keyed by node identity.
    /// Fetched from RPC when not set; RPC only serves the current epoch and the four before it.
    #[arg(long)]
    leader_schedule: Option<PathBuf>,
}

/// Maximum number of accounts per `get_multiple_accounts` RPC call.
const GET_MULTIPLE_ACCOUNTS_BATCH_SIZE: usize = 100;

pub async fn run(
    args: CrankUpdateLeaderSlots,
    rpc_url: String,
    program_id: Pubkey,
) -> anyhow::Result<()> {
    let keypair = read_keypair_file(args.keypair_path)
        .map_err(|e| anyhow!("Failed reading keypair file: {e}"))?;
    let keypair = Arc::new(keypair);
    let client = RpcClient::new(rpc_url.clone());
    let client = Arc::new(client);

    let epoch_info = client.get_epoch_info().await?;
    let epoch = args.epoch.unwrap_or(epoch_info.epoch);
    if epoch > epoch_info.epoch {
        return Err(anyhow!(
            "Cannot write epoch {epoch} because the current epoch is {}",
            epoch_info.epoch
        ));
    }

    println!("Target epoch: {epoch}");

    let epoch_schedule = client.get_epoch_schedule().await?;
    let leader_schedule: HashMap<String, Vec<usize>> = match args.leader_schedule {
        Some(path) => serde_json::from_reader(File::open(path)?)?,
        None => client
            .get_leader_schedule(Some(epoch_schedule.get_first_slot_in_epoch(epoch)))
            .await?
            .ok_or_else(|| anyhow!("Leader schedule unavailable for epoch {epoch}"))?,
    };

    // Fetch validator history accounts
    let validator_histories = get_all_validator_history_accounts(&client, program_id).await?;

    // Collect all vote accounts from validator history
    let candidates: Vec<Pubkey> = validator_histories
        .iter()
        .map(|vh| vh.vote_account)
        .collect();

    println!("Found {} candidate vote accounts", candidates.len());

    // Filter out vote accounts that no longer exist on-chain (closed/reassigned)
    // to avoid ConstraintOwner errors
    let mut live_vote_accounts: HashSet<Pubkey> = HashSet::new();
    for chunk in candidates.chunks(GET_MULTIPLE_ACCOUNTS_BATCH_SIZE) {
        let accounts = client.get_multiple_accounts(chunk).await?;
        for (pubkey, account) in chunk.iter().zip(accounts.iter()) {
            if is_live_vote_account(account.as_ref()) {
                live_vote_accounts.insert(*pubkey);
            }
        }
    }
    let live_vote_accounts: Vec<Pubkey> = live_vote_accounts.into_iter().collect();

    let vote_accounts = get_vote_accounts_with_retry(&client, MIN_VOTE_EPOCHS, None)
        .await?
        .into_iter()
        .map(|vote_account_info| {
            Ok((
                Pubkey::from_str(&vote_account_info.vote_pubkey)?,
                vote_account_info,
            ))
        })
        .collect::<anyhow::Result<HashMap<_, _>>>()?;

    let leader_slots_by_vote = leader_slots_by_vote_account(
        &leader_schedule,
        &vote_accounts,
        &live_vote_accounts,
        epoch_schedule.get_slots_in_epoch(epoch),
    )
    .map_err(|e| anyhow!(e))?;

    // Build instructions for each validator whose count isn't recorded yet
    let validator_histories: HashMap<Pubkey, _> = validator_histories
        .iter()
        .map(|vh| (vh.vote_account, vh))
        .collect();
    let instructions: Vec<_> = leader_slots_by_vote
        .iter()
        .filter(|(vote_account, &total_leader_slots)| {
            needs_leader_slots_update(
                validator_histories.get(vote_account).copied(),
                epoch,
                epoch_info.epoch,
                total_leader_slots,
            )
        })
        .map(|(vote_account, &total_leader_slots)| {
            LeaderSlotsEntry::new(
                *vote_account,
                &program_id,
                &keypair.pubkey(),
                epoch,
                total_leader_slots,
            )
            .update_instruction()
        })
        .collect();

    println!(
        "{} live vote accounts, {} attributed, {} to update",
        live_vote_accounts.len(),
        leader_slots_by_vote.len(),
        instructions.len()
    );

    if instructions.is_empty() {
        println!("All accounts already up to date");
        return Ok(());
    }

    // Use one leader slots update per transaction so individual failures stay isolated.
    let total_txs = instructions.len();
    println!("Sending {total_txs} transactions concurrently...");

    // Send all transactions concurrently in batches of 20,
    // refreshing the blockhash each batch to avoid expiration.
    let mut success_count = 0u64;
    let mut error_count = 0u64;
    let concurrent_batch_size = 20;

    for (batch_idx, ix_batch) in instructions.chunks(concurrent_batch_size).enumerate() {
        let recent_blockhash = client.get_latest_blockhash().await?;

        let futures: Vec<_> = ix_batch
            .iter()
            .enumerate()
            .map(|(i, ix)| {
                let tx_idx = batch_idx * concurrent_batch_size + i + 1;
                let client = client.clone();
                let tx = Transaction::new_signed_with_payer(
                    std::slice::from_ref(ix),
                    Some(&keypair.pubkey()),
                    &[&*keypair],
                    recent_blockhash,
                );
                async move {
                    match client.send_transaction(&tx).await {
                        Ok(sig) => {
                            println!("[{tx_idx}/{total_txs}] Sent: {sig}");
                            Ok(())
                        }
                        Err(e) => {
                            println!("[{tx_idx}/{total_txs}] Error: {e}");
                            Err(e)
                        }
                    }
                }
            })
            .collect();

        let results = join_all(futures).await;
        for result in results {
            match result {
                Ok(()) => success_count += 1,
                Err(_) => error_count += 1,
            }
        }
    }

    println!(
        "Done. {} sent, {} failed out of {} transactions ({} instructions)",
        success_count,
        error_count,
        total_txs,
        instructions.len()
    );

    Ok(())
}
