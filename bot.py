import os
import sys
import time
import random
import warnings
import requests
import numpy as np
from datetime import datetime, timedelta, timezone

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import yt_dlp
from apify_client import ApifyClient
import cv2
import torch
import gspread
from PIL import Image
from torchvision import transforms
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# ==========================================
# ⚙️ เปิด-ปิด ระบบ Test Message ส่งเข้า Google Chat ทุกรอบ
ENABLE_STATUS_MESSAGE = True
# ==========================================

# --- การตั้งค่า API และ Webhook ---
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY")
APIFY_TOKEN = os.environ.get("APIFY_TOKEN")
SHEET_ID = os.environ.get("SHEET_ID")
GDRIVE_FOLDER_ID = os.environ.get("GDRIVE_FOLDER_ID")

# 🔴 ลิงก์ Webhook Google Chat ของคุณ
GOOGLE_CHAT_WEBHOOK = "https://chat.googleapis.com/v1/spaces/AAQAGsvHT0c/messages?key=AIzaSyDdI0hCZtE6vySjMm-WEfRq3CPzqKqqsHI&token=_gjfX3kZs7NEU6fxNYYTvVkhZFEC7WkwfEdxZ0fvKTw"

BASE_PATH = './'
MODEL_PATH = os.path.join(BASE_PATH, 'bigc_model.pth')
BKK_TZ = timezone(timedelta(hours=7))
device = torch.device("cpu")

# --- Authentication (Sheet & Drive) ---
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
try:
    creds = Credentials.from_service_account_file('credentials.json', scopes=SCOPES)
    gc = gspread.authorize(creds)
    drive_service = build('drive', 'v3', credentials=creds)
except Exception as e:
    print(f"❌ Error Auth: {e}")
    sys.exit(1)

def get_bkk_now(): return datetime.now(BKK_TZ)

def format_to_bkk(date_input):
    try:
        if isinstance(date_input, (int, float)):
            val = date_input if date_input < 1e11 else date_input / 1000.0
            dt = datetime.fromtimestamp(val, timezone.utc)
        else:
            clean_str = str(date_input).replace('Z', '+00:00').replace("'", "").strip()
            dt = datetime.fromisoformat(clean_str[:19] + '+00:00')
        return dt.astimezone(BKK_TZ).strftime('%Y-%m-%d %H:%M:%S')
    except: return str(date_input).replace("'", "").strip()

def send_google_chat_message(message):
    """ส่งข้อความเข้า Google Chat ผ่าน Webhook"""
    if not GOOGLE_CHAT_WEBHOOK: 
        print("⚠️ GOOGLE_CHAT_WEBHOOK is missing!")
        return
    try:
        res = requests.post(
            GOOGLE_CHAT_WEBHOOK,
            headers={"Content-Type": "application/json"},
            json={"text": message}
        )
        if res.status_code != 200:
            print(f"⚠️ Google Chat API Error: {res.text}")
        else:
            print("✅ ส่ง Google Chat สำเร็จ!")
    except Exception as e: 
        print(f"⚠️ Google Chat Send Exception: {e}")

def update_heartbeat(ws_control):
    try: ws_control.update_cell(9, 2, get_bkk_now().strftime('%Y-%m-%d %H:%M:%S'))
    except: pass

def load_ai_model():
    if os.path.exists(MODEL_PATH):
        m = torch.load(MODEL_PATH, map_location=device, weights_only=False)
        if hasattr(m, 'eval'): m.eval()
        print("🧠 AI Model: ONLINE")
        return m
    print("⚠️ Model not found")
    return None

def predict_logo(model, frame):
    if model is None: return False, 0.0, None
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(cv2.resize(rgb, (224, 224)))
        transform = transforms.Compose([
            transforms.Resize((224, 224)), transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        with torch.no_grad():
            output = model(transform(pil_img).unsqueeze(0).to(device))
            prob = torch.nn.functional.softmax(output[0], dim=0)
            conf, pred = torch.max(prob, 0)
        if pred.item() == 0 and conf.item() > 0.85: return True, conf.item(), Image.fromarray(rgb)
    except: pass
    return False, 0.0, None

def save_evidence(image_pil, video_id, timestamp_str):
    """อัปโหลดรูปลง Google Drive ผ่าน API"""
    try:
        safe_ts = str(timestamp_str).replace(":", "_")
        local_filename = f"DETECT_{video_id}_{safe_ts}.jpg"
        image_pil.save(local_filename)

        if not GDRIVE_FOLDER_ID:
            print("⚠️ ไม่พบ GDRIVE_FOLDER_ID (ยังไม่ได้ตั้งค่าใน Secrets)")
            return "-", "-"

        file_metadata = {'name': local_filename, 'parents': [GDRIVE_FOLDER_ID]}
        media = MediaFileUpload(local_filename, mimetype='image/jpeg', resumable=True)
        file = drive_service.files().create(body=file_metadata, media_body=media, fields='id, webViewLink').execute()
        
        drive_service.permissions().create(fileId=file.get('id'), body={'type': 'anyone', 'role': 'reader'}).execute()
        
        url = file.get('webViewLink')
        os.remove(local_filename)
        
        return f'=HYPERLINK("{url}", "🖼️ กดดูรูปภาพ")', url
    except Exception as e: 
        print(f" ⚠️ Upload Error: {e}")
        return "-", "-"

def fetch_data(platforms, keywords, max_res, days_back):
    all_videos = []
    client = ApifyClient(APIFY_TOKEN)
    cutoff_utc = datetime.now(timezone.utc) - timedelta(days=days_back)
    
    for plat in platforms:
        for kw in keywords:
            print(f"📡 Searching '{kw}' on {plat}...")
            try:
                if plat == 'YouTube':
                    params = {'part': 'snippet', 'q': kw, 'key': YOUTUBE_API_KEY, 'maxResults': max_res, 'type': 'video', 'publishedAfter': cutoff_utc.isoformat().replace('+00:00', 'Z')}
                    res = requests.get("https://www.googleapis.com/youtube/v3/search", params=params).json()
                    for item in res.get('items', []):
                        all_videos.append({
                            'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}",
                            'title': item['snippet']['title'], 'platform': 'YouTube',
                            'user': item['snippet']['channelTitle'], 'date': format_to_bkk(item['snippet']['publishedAt']),
                            'image_url': item['snippet']['thumbnails']['high']['url'], 'id': item['id']['videoId']
                        })
                else:
                    actor = {'TikTok': 'clockworks/tiktok-scraper', 'Instagram': 'apify/instagram-hashtag-scraper', 'Facebook': 'apify/facebook-search-scraper'}.get(plat)
                    if not actor: continue
                    inp = {"resultsLimit": max_res, "maxItems": max_res, "resultsPerPage": max_res, "proxyConfiguration": {"useApifyProxy": True}}
                    if plat in ['TikTok', 'Instagram']: inp["hashtags"] = [kw.replace(" ","")]
                    else: inp["searchTerms"] = kw
                    
                    run = client.actor(actor).call(run_input=inp, timeout_secs=90)
                    for item in client.dataset(run["defaultDatasetId"]).list_items().items:
                        raw_date = item.get('createTime') or item.get('timestamp') or item.get('date')
                        try:
                            if isinstance(raw_date, (int, float)): dt_utc = datetime.fromtimestamp(raw_date if raw_date < 1e11 else raw_date/1000.0, timezone.utc)
                            else: dt_utc = datetime.fromisoformat(str(raw_date).replace('Z', '+00:00')[:19] + '+00:00')
                            if dt_utc < cutoff_utc: continue 
                        except: pass

                        user = item.get('authorNickname') or item.get('authorName') or item.get('ownerUsername') or "Unknown"
                        v_url = item.get('webVideoUrl') or item.get('videoWebUrl') or item.get('url') or item.get('postUrl')
                        if not v_url and plat == 'TikTok' and item.get('id'): v_url = f"https://www.tiktok.com/@{user}/video/{item.get('id')}"
                            
                        if v_url:
                            all_videos.append({
                                'url': v_url, 'title': (item.get('text') or item.get('desc') or "No Title")[:200], 'platform': plat, 'user': user,
                                'date': format_to_bkk(raw_date), 'id': str(item.get('id', random.randint(1,9999))),
                                'image_url': item.get('displayUrl') or item.get('imageUrl') or item.get('coverUrl')
                            })
            except Exception as e: print(f" ⚠️ {plat} Fetch Error: {e}")
    return all_videos

def main():
    sh = gc.open_by_key(SHEET_ID)
    ws_data = sh.worksheet("Apify")
    ws_control = sh.worksheet("Control_Panel")
    try: ws_logs = sh.worksheet("Scan_Logs")
    except: ws_logs = sh.add_worksheet(title="Scan_Logs", rows=1000, cols=4)

    update_heartbeat(ws_control)
    config = ws_control.col_values(2)
    status = str(config[0]).strip() if len(config) > 0 else "🔴 Stop"

    if 'Start' not in status and '🟢' not in status:
        print("🛑 Status is Stop. Exiting gracefully.")
        sys.exit(0)

    platforms = [p.strip() for p in config[1].split(',')]
    keywords = [k.strip() for k in config[2].split(',')]
    days_back = 1 if '1' in str(config[3]) else 7 if '2' in str(config[3]) else 30
    max_res = int(config[6]) if str(config[6]).isdigit() else 5
    
    print(f"\n🚀 Serverless Mode Activated: {get_bkk_now().strftime('%H:%M:%S')}")
    raw_list = fetch_data(platforms, keywords, max_res, days_back)
    processed_urls = set(ws_data.col_values(7)[1:]) 
    
    unique_list, duplicate_list, seen = [], [], set()
    for v in raw_list:
        if v['url'] in processed_urls or v['url'] in seen: duplicate_list.append(v)
        else: seen.add(v['url']); unique_list.append(v)
    
    print(f"📋 Found: {len(raw_list)} (New: {len(unique_list)} / Dup: {len(duplicate_list)})")
    
    # --- 1. บันทึก Log รายละเอียด ---
    print("📋 กำลังบันทึก Log รายละเอียด...")
    new_text = "\n".join([f"[{u['platform']}] {str(u['title']).replace(chr(10), ' ')[:50]}... -> {u['url']}" for u in unique_list])
    dup_text = "\n".join([f"[{d['platform']}] {str(d['title']).replace(chr(10), ' ')[:50]}... -> {d['url']}" for d in duplicate_list])

    try: 
        ws_logs.insert_row([
            get_bkk_now().strftime('%Y-%m-%d %H:%M:%S'), 
            dup_text if dup_text else "-", 
            new_text if new_text else "-", 
            ", ".join(platforms)
        ], index=2)
    except Exception as e:
        print(f"⚠️ บันทึก Log ไม่สำเร็จ: {e}")

    engine = load_ai_model()
    chat_summary = []

    # --- 2. สแกนหาโลโก้ ---
    for v in unique_list:
        print(f"👁️ Scanning: [{v['user']}] - {v['title'][:40]}...")
        found, final_ts, best_img = False, "-", None

        try:
            with yt_dlp.YoutubeDL({'format': 'best', 'quiet': True}) as ydl:
                stream = ydl.extract_info(v['url'], download=False).get('url')
                if stream:
                    cap = cv2.VideoCapture(stream)
                    for i in range(150): 
                        ret, frame = cap.read()
                        if not ret: break
                        if i % 30 == 0:
                            hit, sc, img = predict_logo(engine, frame)
                            if hit:
                                found, best_img, final_ts = True, img, f"{int((i//30)//60):02d}:{int((i//30)%60):02d}"
                                break
                    cap.release()
        except: pass

        if not found and v.get('image_url'):
            try:
                resp = requests.get(v['image_url'], timeout=10)
                img = cv2.imdecode(np.asarray(bytearray(resp.content), dtype=np.uint8), cv2.IMREAD_COLOR)
                hit, sc, img_res = predict_logo(engine, img)
                if hit: found, best_img, final_ts = True, img_res, "Image"
            except: pass

        if found:
            print(f"  🎯 Logo Found!")
            formula, img_url = save_evidence(best_img, v['id'], final_ts)
            clean_title = ("'" + v['title']) if str(v['title']).startswith(('=', '+', '-')) else v['title'].replace('\n', ' ')
            try: ws_data.insert_row([v['date'], clean_title, v['platform'], v['user'], "Yes", final_ts, v['url'], formula], index=2, value_input_option='USER_ENTERED')
            except: pass
            
            chat_summary.append(f"[{v['platform']}] {v['user']}\n🎬 {clean_title[:30]}...\n🔗 ลิงก์: {v['url']}\n🖼️ รูป: {img_url}")
        else: print("  ❌ No logo.")

    # --- 3. ส่ง Google Chat (Alert) ถ้าเจอโลโก้ ---
    if chat_summary:
        print(f"📱 Sending batched Google Chat message ({len(chat_summary)} items)...")
        display_list = chat_summary[:10]
        final_msg = f"🚨 แจ้งเตือน! พบโลโก้ใหม่ {len(chat_summary)} รายการ:\n" + "="*20 + "\n"
        final_msg += "\n\n".join(display_list)
        if len(chat_summary) > 10:
            final_msg += f"\n\n... และอื่นๆ อีก {len(chat_summary) - 10} รายการ\n(ดูต่อใน Google Sheets)"
        send_google_chat_message(final_msg)

    # --- 4. 🧪 ส่ง Test Message สรุปสถานะ ---
    if ENABLE_STATUS_MESSAGE and len(raw_list) > 0:
        latest = raw_list[0]
        test_msg = (
            f"🤖 [System Test: GitHub Bot]\n"
            f"เวลา: {get_bkk_now().strftime('%H:%M:%S')}\n"
            f"ดึงข้อมูลทั้งหมด: {len(raw_list)} คลิป\n"
            f"คัดกรองคลิปใหม่: {len(unique_list)} คลิป\n"
            f"เจอโลโก้รอบนี้: {len(chat_summary)} คลิป\n"
            f"{'='*20}\n"
            f"📌 ตัวอย่างข้อมูลล่าสุดที่ดูดมาได้:\n"
            f"[{latest['platform']}] {latest['user']}\n"
            f"🎬 {str(latest['title']).replace(chr(10), ' ')[:40]}...\n"
            f"🔗 {latest['url']}"
        )
        send_google_chat_message(test_msg)

    if 'Run Once' in str(config[4]): ws_control.update_cell(1, 2, '🔴 Stop')
    print("✅ Run Complete. Serverless container will now self-destruct.")

if __name__ == "__main__":
    main()
