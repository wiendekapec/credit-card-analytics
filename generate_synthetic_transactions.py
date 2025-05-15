"""
generate_synthetic_transactions.py
Синтетический набор транзакций для BI‑демо‑проекта.

Колонки
-------
txn_id           – уникальный id операции
customer_id      – id клиента (0‑based)
timestamp        – datetime операции (2019‑01‑01 … 2019‑12‑31)
amount           – сумма транзакции, руб
mcc_code         – 4‑значный MCC (символ категорий)
is_fraud         – 0/1 (≈ 0.2 % мошеннических)
"""
import numpy as np
import pandas as pd

# ─── параметры ───────────────────────────────────────────────────
N_CUSTOMERS   = 50_000
AVG_TXNS_CUST = 20          # среднее число операций на клиента
YEAR          = 2019
FRAUD_RATE    = 0.002       # 0.2 %

# ─── 1. сколько транзакций у каждого клиента ────────────────────
rng          = np.random.default_rng(42)
txns_per_cus = rng.poisson(lam=AVG_TXNS_CUST, size=N_CUSTOMERS)
txns_per_cus[ txns_per_cus == 0 ] = 1   # никто не остаётся «пустым»
N_TXN        = txns_per_cus.sum()

# ─── 2. customer_id для каждой строки ───────────────────────────
customer_id  = np.repeat(np.arange(N_CUSTOMERS), txns_per_cus)

# ─── 3. даты / время (равномерно на год) ─────────────────────────
start = pd.Timestamp(f'{YEAR}-01-01')
end   = pd.Timestamp(f'{YEAR}-12-31 23:59:59')
seconds_range = int((end - start).total_seconds())
timestamp = start + pd.to_timedelta(rng.integers(0, seconds_range, N_TXN), unit='s')

# ─── 4. суммы операций (логнормальное распределение) ────────────
amount = rng.lognormal(mean=3.5, sigma=0.9, size=N_TXN).round(2)  # ≈ руб 30‑40 000 p95

# ─── 5. MCC (берём 15 популярных категорий) ─────────────────────
MCCs   = [5411, 5814, 5912, 5999, 4829, 5732, 5651, 4111, 5944,
          4812, 5699, 6011, 5541, 5941, 5311]
mcc_code = rng.choice(MCCs, size=N_TXN, p=np.full(len(MCCs), 1/len(MCCs)))

# ─── 6. признак fraude ───────────────────────────────────────────
is_fraud = rng.choice([0, 1], size=N_TXN, p=[1-FRAUD_RATE, FRAUD_RATE])

# ─── 7. собираем DataFrame ──────────────────────────────────────
df = pd.DataFrame({
    'txn_id'      : np.arange(N_TXN),
    'customer_id' : customer_id,
    'timestamp'   : timestamp,
    'amount'      : amount,
    'mcc_code'    : mcc_code,
    'is_fraud'    : is_fraud
})

# ─── 8. сохраняем CSV ───────────────────────────────────────────
csv_path = "creditcard_synth.csv"
df.to_csv(csv_path, index=False)
print(f"✔ Synthetic dataset saved → {csv_path}\n   {N_TXN:,} rows, {N_CUSTOMERS:,} customers.")
