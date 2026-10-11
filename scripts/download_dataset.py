from pathlib import Path
from urllib.request import Request, urlopen

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"

def main():
    out = Path("data/raw/Telco-Customer-Churn.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    request = Request(URL, headers={"User-Agent":"ChurnIQ educational project"})
    try:
        with urlopen(request, timeout=45) as response:
            content = response.read()
    except Exception as exc:
        raise SystemExit(f"Download failed: {exc}. Download the CSV manually to {out}.")
    if b"customerID" not in content[:5000] or b"Churn" not in content[:5000]:
        raise SystemExit("Downloaded file did not look like the expected CSV.")
    out.write_bytes(content)
    print(f"Saved {len(content):,} bytes to {out}")

if __name__ == "__main__":
    main()
