-- How concentrated are fraud losses? Split fraud transactions into 10 equal-size
-- groups by amount (decile 10 = largest) and show each decile's share of total
-- fraud dollars, plus a running total.
WITH fraud AS (
    SELECT
        Amount,
        NTILE(10) OVER (ORDER BY Amount) AS decile
    FROM transactions
    WHERE Class = 1
),
by_decile AS (
    SELECT
        decile,
        COUNT(*)              AS frauds,
        ROUND(MIN(Amount), 2) AS min_amount,
        ROUND(MAX(Amount), 2) AS max_amount,
        SUM(Amount)           AS decile_amount
    FROM fraud
    GROUP BY decile
)
SELECT
    decile,
    frauds,
    min_amount,
    max_amount,
    ROUND(100.0 * decile_amount / SUM(decile_amount) OVER (), 1) AS share_of_fraud_dollars_pct,
    ROUND(100.0 * SUM(decile_amount) OVER (ORDER BY decile DESC)
          / SUM(decile_amount) OVER (), 1)                       AS cumulative_from_top_pct
FROM by_decile
ORDER BY decile DESC;
