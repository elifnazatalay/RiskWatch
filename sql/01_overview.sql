-- Portfolio overview: how much fraud is there, by count and by dollar value?
SELECT
    COUNT(*)                                          AS total_transactions,
    SUM(Class)                                        AS fraud_transactions,
    ROUND(100.0 * SUM(Class) / COUNT(*), 3)           AS fraud_rate_pct,
    ROUND(SUM(Amount), 2)                             AS total_amount,
    ROUND(SUM(CASE WHEN Class = 1 THEN Amount END), 2) AS fraud_amount,
    ROUND(100.0 * SUM(CASE WHEN Class = 1 THEN Amount END) / SUM(Amount), 3)
                                                      AS fraud_amount_pct,
    ROUND(AVG(CASE WHEN Class = 0 THEN Amount END), 2) AS avg_legit_amount,
    ROUND(AVG(CASE WHEN Class = 1 THEN Amount END), 2) AS avg_fraud_amount
FROM transactions;
