#!/usr/bin/env python3
import sage_data_client
import json


df = sage_data_client.query("-15m", filter={"name": "inference_log"})
df["vsn"] = df["meta.vsn"]

for r in df.itertuples():
    inference = json.loads(r.value)
    inference["timestamp"] = r.timestamp.isoformat()
    inference["vsn"] = r.vsn
    print(json.dumps(inference, separators=(",", ":"), sort_keys=True))
