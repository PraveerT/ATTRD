#!/usr/bin/env python3
import os
import time
import requests
import re
from datetime import datetime

# Telegram Bot Configuration
BOT_TOKEN = "8049556095:AAH0c0KB0DmzFtcW0s97ZS_kQ8ux9gX72eE"
CHAT_ID = None  # Will be set on first run
LOG_FILE = "/notebooks/PMamba/experiments/work_dir/baseline/train.txt"
STATE_FILE = "/notebooks/PMamba/experiments/.telegram_monitor_state"

def get_chat_id():
    """Get chat ID from the first message to the bot"""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/getUpdates"
    response = requests.get(url)
    data = response.json()
    
    if data["ok"] and data["result"]:
        # Get the most recent message
        chat_id = data["result"][-1]["message"]["chat"]["id"]
        return chat_id
    return None

def send_telegram_message(message):
    """Send message to Telegram"""
    global CHAT_ID
    
    if CHAT_ID is None:
        CHAT_ID = get_chat_id()
        if CHAT_ID is None:
            print("No chat ID found. Please send /start to your bot first.")
            return False
    
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    data = {
        "chat_id": CHAT_ID,
        "text": message,
        "parse_mode": "HTML"
    }
    
    try:
        response = requests.post(url, data=data)
        return response.json()["ok"]
    except:
        return False

def load_state():
    """Load previous best accuracy, chat ID, and last processed line"""
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, 'r') as f:
            lines = f.readlines()
            best_acc = float(lines[0].strip()) if lines else 0.0
            chat_id = int(lines[1].strip()) if len(lines) > 1 and lines[1].strip() else None
            last_line = int(lines[2].strip()) if len(lines) > 2 and lines[2].strip() else 0
            return best_acc, chat_id, last_line
    return 0.0, None, 0

def save_state(best_acc, chat_id, last_line):
    """Save current best accuracy, chat ID, and last processed line"""
    with open(STATE_FILE, 'w') as f:
        f.write(f"{best_acc}\n")
        if chat_id:
            f.write(f"{chat_id}\n")
        f.write(f"{last_line}\n")

def parse_log_file(start_line=0):
    """Parse training log and extract new entries"""
    if not os.path.exists(LOG_FILE):
        return [], None
    
    with open(LOG_FILE, 'r') as f:
        lines = f.readlines()
    
    new_entries = []
    best_acc = 0.0
    best_epoch = None
    
    # Parse each line for accuracy
    pattern = r'Epoch (\d+), Test, Evaluation: prec1 ([\d.]+), prec5 ([\d.]+)'
    
    for i, line in enumerate(lines):
        match = re.search(pattern, line)
        if match:
            epoch = int(match.group(1))
            acc = float(match.group(2))
            prec5 = float(match.group(3))
            
            if i >= start_line:
                # This is a new entry
                timestamp = re.search(r'\[ (.+?) \]', line)
                time_str = timestamp.group(1) if timestamp else ""
                new_entries.append((i, epoch, acc, prec5, time_str))
            
            if acc > best_acc:
                best_acc = acc
                best_epoch = epoch
    
    return new_entries, (best_epoch, best_acc)

def format_new_entry_message(entry, best, is_new_best=False):
    """Format message for new training entry"""
    line_num, epoch, acc, prec5, time_str = entry
    
    msg = f"📊 <b>Epoch {epoch} Complete</b>\n"
    msg += f"├ Top-1 Acc: {acc:.2f}%\n"
    msg += f"├ Top-5 Acc: {prec5:.2f}%\n"
    
    if is_new_best:
        msg += f"🏆 <b>New Best!</b>\n"
    
    if best and best[1] > 0:
        msg += f"└ Best so far: {best[1]:.2f}% (Epoch {best[0]})\n"
    
    return msg

def main():
    global CHAT_ID
    
    print("🚀 Starting Telegram Training Monitor")
    print(f"📁 Monitoring: {LOG_FILE}")
    
    # Load previous state
    prev_best, CHAT_ID, last_epoch = load_state()
    print(f"📈 Previous best accuracy: {prev_best:.2f}%")
    print(f"📍 Last sent epoch: {last_epoch}")
    
    if CHAT_ID:
        print(f"💬 Using saved chat ID: {CHAT_ID}")
    else:
        print("⚠️  No chat ID saved. Send /start to your bot to initialize.")
    
    # Send initial message
    send_telegram_message("🚀 Training monitor started!\nI'll notify you when new epochs complete.")
    
    check_interval = 30  # Check every 30 seconds for new entries
    
    while True:
        try:
            # Parse entire log to find latest entry
            all_entries, best = parse_log_file(0)
            
            if all_entries:
                # Get the latest entry
                latest_entry = all_entries[-1]
                line_num, epoch, acc, prec5, time_str = latest_entry
                
                # Only send if this is a new epoch we haven't sent yet
                if epoch > last_epoch:
                    # Check if this is a new best
                    is_new_best = acc > prev_best
                    if is_new_best:
                        prev_best = acc
                    
                    # Send message for this new entry
                    message = format_new_entry_message(latest_entry, best, is_new_best)
                    if send_telegram_message(message):
                        print(f"✅ Sent: Epoch {epoch}, Acc: {acc:.2f}%{' (NEW BEST!)' if is_new_best else ''}")
                    
                    # Update last sent epoch
                    last_epoch = epoch
                    save_state(prev_best, CHAT_ID, last_epoch)
            
            time.sleep(check_interval)
            
        except KeyboardInterrupt:
            print("\n👋 Stopping monitor...")
            send_telegram_message("🛑 Training monitor stopped.")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            time.sleep(check_interval)

if __name__ == "__main__":
    main()