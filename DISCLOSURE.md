# Vendor disclosure workflow
1. Run only against models you own or are authorised to test; read the provider's usage policy / red-team programme first.
2. Promote confirmed jailbreaks to the local corpus (Live Console or DB Explorer). Re-verify before reporting.
3. Corpus page → select items for **one model** → *Build vendor report*. The report is hashed + HMAC-signed and an audit row is written (`disclosures.hold_until` = generated + 90 days by default).
4. Send the report to the vendor's security contact yourself — nothing is transmitted automatically.
5. Publish only aggregate rates (leaderboard/report) before the hold expires; raw responses stay local (`--redact-responses` default). Re-verify after vendor updates: `working → patched`.
Contact for this project: security@example.org (placeholder).
