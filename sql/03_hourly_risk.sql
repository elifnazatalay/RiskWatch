-- Fraud rate by hour, ranked from riskiest to safest.
-- Note: Time is seconds since the first transaction in the dataset, not a real
-- clock time, so "hour 0" means the first hour of data collection (the data
-- covers two days, so each hour bucket combines both days).
WITH hourly AS (
    SELECT
        CAST(Time / 3600 AS INTEGER) % 24 AS hour,
        COUNT(*)                          AS transactions,
        SUM(Class)                        AS frauds
    FROM transactions
    GROUP BY hour
)
SELECT
    hour,
    transactions,
    frauds,
    ROUND(100.0 * frauds / transactions, 3) AS fraud_rate_pct,
    RANK() OVER (ORDER BY 1.0 * frauds / transactions DESC) AS risk_rank,
    -- Share of the day's volume that falls in this hour (low-volume hours are often riskier).
    ROUND(100.0 * transactions / SUM(transactions) OVER (), 2) AS volume_share_pct
FROM hourly
ORDER BY risk_rank;
