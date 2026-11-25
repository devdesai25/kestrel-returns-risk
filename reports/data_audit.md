# Data audit

## Duplicates
- Partner-feed re-imports: 651 order_ids appeared twice.
- Kept `crm` copy -> 10504 rows (base return rate 11.42%).

## October 2025 value inflation
- Rows: 700
- Before mean/max: 783798 / 5880204
- After mean/max: 7838 / 58802
- Other months mean/max: 8071 / 58802
- Fix: Divided October-2025 order_value_inr by 100 (100x gateway inflation).

## Leakage
last_service_event_type==REVERSE_PICKUP and pickup_scheduled_at are post-return artefacts (ops-policy section 7). Dropped from the feature set.
- With leakage AUC/PR: {'auc_roc': 0.9955147609282947, 'auc_pr': 0.9833734720128988}
- Without leakage AUC/PR: {'auc_roc': 0.7621109049680479, 'auc_pr': 0.3834014054840814}

## Service-state mismatch (train vs test)
Train reflects as-of-export-day service state (includes DEMO_DONE / INSTALL_DONE / REVERSE_PICKUP). Test is at-dispatch (NONE / INSTALL_BOOKED). INSTALL_BOOKED never appears in train. Column cannot be used across files.
- Train: {'NONE': 0.6222, 'DEMO_DONE': 0.1349, 'INSTALL_DONE': 0.1289, 'REVERSE_PICKUP': 0.0714, 'TECH_VISIT': 0.0426}
- Test: {'NONE': 0.7409, 'INSTALL_BOOKED': 0.2591}

## Pincode 000000
delivery_pincode==000000 appears across channels (not only partner_outlet); treated as missing -> pin_missing flag.

## Delivery notes
Free-text `delivery_note` carries almost no signal; only `note_missing` is kept.
