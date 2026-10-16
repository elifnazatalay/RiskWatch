-- Fraud rate by transaction-amount band.
-- lift = band fraud rate / overall fraud rate (lift 2.0 = twice as risky as average).
WITH banded AS (
    SELECT
        Class,
        Amount,
        CASE
            WHEN Amount = 0     THEN '0. $0'
            WHEN Amount < 10    THEN '1. $0-10'
            WHEN Amount < 50    THEN '2. $10-50'
            WHEN Amount < 100   THEN '3. $50-100'
            WHEN Amount < 500   THEN '4. $100-500'
            WHEN Amount < 1000  THEN '5. $500-1,000'
            ELSE                     '6. $1,000+'
        END AS amount_band
    FROM transactions
),
overall AS (
    SELECT 1.0 * SUM(Class) / COUNT(*) AS rate FROM transactions
)
SELECT
    amount_band,
    COUNT(*)                                      AS transactions,
    SUM(Class)                                    AS frauds,
    ROUND(100.0 * SUM(Class) / COUNT(*), 3)       AS fraud_rate_pct,
    ROUND((1.0 * SUM(Class) / COUNT(*)) / overall.rate, 2) AS lift,
    ROUND(100.0 * SUM(Class) / SUM(SUM(Class)) OVER (), 1) AS share_of_all_fraud_pct
FROM banded, overall
GROUP BY amount_band
ORDER BY amount_band;
