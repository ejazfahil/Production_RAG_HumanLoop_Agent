# ACME X200 — Field Maintenance Manual (Excerpt)

## Section 4: Routine Inspection

Field technicians should inspect the meter terminal cover seal at every site visit.
A broken or missing seal must be reported as a tamper event within 24 hours.

## Section 5: Battery Replacement

The X200 contains a non-rechargeable lithium backup battery that powers the real-time
clock during outages. Expected battery life is 10 years. When the low-battery flag is
raised, schedule replacement within 30 days. Use only ACME part number BT-3V6-AA.

## Section 6: Communication Troubleshooting

If the meter does not respond over DLMS, verify the RS-485 wiring polarity first.
A swapped A/B pair is the most common cause of no-response faults. If polarity is
correct, confirm the baud rate matches the head-end configuration (default 19200).

## Section 7: Replacement Procedure

Before removing a meter, record the final energy register reading and the event log.
De-energize the supply, remove the terminal cover, and disconnect in the order:
neutral last. The replacement meter must be commissioned with the same DLMS client
address as the unit it replaces.
