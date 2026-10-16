-- Compare day 1 vs day 2 of the dataset. Big changes between periods are an
-- early sign of drift, which matters later when monitoring the model.
SELECT
    CAST(Time / 86400 AS INTEGER) + 1           AS day,
    COUNT(*)                                    AS transactions,
    SUM(Class)                                  AS frauds,
    ROUND(100.0 * SUM(Class) / COUNT(*), 3)     AS fraud_rate_pct,
    ROUND(AVG(Amount), 2)                       AS avg_amount,
    ROUND(AVG(CASE WHEN Class = 1 THEN Amount END), 2) AS avg_fraud_amount,
    ROUND(100.0 * (1.0 * SUM(Class) / COUNT(*))
          / LAG(1.0 * SUM(Class) / COUNT(*)) OVER (ORDER BY CAST(Time / 86400 AS INTEGER)) - 100, 1)
                                                AS fraud_rate_change_vs_prev_day_pct
FROM transactions
GROUP BY day
ORDER BY day;
