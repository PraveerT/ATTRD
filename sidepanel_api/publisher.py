"""Push status JSON to the Railway endpoint every N seconds.

Usage:
  ANEMON_PUBLISH_TOKEN=... ANEMON_PUBLISH_URL=https://.../api/anemon-publish \
    python3 publisher.py [--interval 30]

Token+URL can also be read from /notebooks/PMamba/sidepanel_api/state/publisher.env
as KEY=VALUE lines.
"""
import argparse, json, os, sys, time, urllib.request, urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server import build_status  # noqa: E402

ENV_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'state', 'publisher.env')


def load_env_file():
    if not os.path.isfile(ENV_FILE):
        return
    with open(ENV_FILE) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#') or '=' not in line:
                continue
            k, v = line.split('=', 1)
            os.environ.setdefault(k.strip(), v.strip())


def post_once(url, token, timeout=10):
    data = json.dumps(build_status()).encode('utf-8')
    req = urllib.request.Request(
        url, data=data, method='POST',
        headers={
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json',
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode('utf-8', errors='replace')[:200]


def main():
    load_env_file()
    p = argparse.ArgumentParser()
    p.add_argument('--interval', type=int, default=30, help='seconds between pushes')
    args = p.parse_args()

    url = os.environ.get('ANEMON_PUBLISH_URL')
    token = os.environ.get('ANEMON_PUBLISH_TOKEN')
    if not url or not token:
        print('ERROR: set ANEMON_PUBLISH_URL and ANEMON_PUBLISH_TOKEN (or write them to state/publisher.env)')
        sys.exit(2)

    print(f'[publisher] target={url} interval={args.interval}s')
    fails = 0
    while True:
        t0 = time.time()
        try:
            code, body = post_once(url, token)
            if code == 200:
                fails = 0
                print(f'[publisher] {time.strftime("%H:%M:%S")} 200 {body[:80]}')
            else:
                fails += 1
                print(f'[publisher] {time.strftime("%H:%M:%S")} HTTP {code}: {body[:120]}')
        except urllib.error.HTTPError as e:
            fails += 1
            print(f'[publisher] {time.strftime("%H:%M:%S")} HTTPError {e.code}: {e.read().decode("utf-8", errors="replace")[:120]}')
        except Exception as e:
            fails += 1
            print(f'[publisher] {time.strftime("%H:%M:%S")} EXC: {type(e).__name__}: {e}')
        # back off on repeated failures (cap at 5 min)
        sleep = args.interval if fails == 0 else min(300, args.interval * (1 + fails))
        elapsed = time.time() - t0
        time.sleep(max(1, sleep - elapsed))


if __name__ == '__main__':
    main()
