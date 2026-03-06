# -*- coding: utf-8 -*-
"""
🚀 INTELLIGENT LOGO HUNTER v7.2 (Stable Core)
- ระบบสลับ YouTube API Key และ Apify Token อัตโนมัติ (รับจาก Secrets)
- อ่านค่า Time Filter แบบเดิม (1=Day, 2=Week, 3=Month)
- เก็บ Log คลิปเก่ารวมในลิสต์ "ซ้ำ" (ฝั่งซ้าย) แต่ไม่สแกนและไม่ลง Sheet Apify
- Batch Insert & Timestamp (คอลัมน์ J)
"""

import os
import sys
import time
import random
import warnings
import subprocess
import requests
import re
import numpy as np
import urllib.parse
from datetime import datetime, timedelta, timezone

warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

ENABLE_STATUS_MESSAGE = True

def install_package(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

needed_packages = ['yt-dlp', 'apify-client', 'gspread', 'opencv-python', 'torch', 'torchvision']
for pkg in needed_packages:
    try: __import__(pkg.replace('-', '_'))
    except ImportError: install_package(pkg)

import yt_dlp
from apify_client import ApifyClient
import cv2
import torch
import gspread
from google.colab import drive, auth
from google.auth import default
from PIL import Image
from torchvision import transforms

# ==========================================
# 🔑 ดึง API Keys จาก Environment (รองรับการใส่หลายคีย์คั่นด้วย ,)
# ==========================================
yt_env = os.environ.get("YOUTUBE_API_KEYS") or os.environ.get("YOUTUBE_API_KEY") or ""
YOUTUBE_API_KEYS = [k.strip() for k in yt_env.split(',') if k.strip()]

apify_env = os.environ.get("APIFY_TOKENS") or os.environ.get("APIFY_TOKEN") or ""
APIFY_TOKENS = [k.strip() for k in apify_env.split(',') if k.strip()]

SHEET_ID = os.environ.get("SHEET_ID")
GDRIVE_FOLDER_ID = os.environ.get("GDRIVE_FOLDER_ID")

# 🔴 WEBHOOK URL
GOOGLE_CHAT_WEBHOOK = "https://chat.googleapis.com/v1/spaces/AAQAGsvHT0c/messages?key=AIzaSyDdI0hCZtE6vySjMm-WEfRq3CPzqKqqsHI&token=_gjfX3kZs7NEU6fxNYYTvVkhZFEC7WkwfEdxZ0fvKTw"
GOOGLE_CHAT_WEBHOOK_TIKTOK = "https://chat.googleapis.com/v1/spaces/AAQAtSkMPnY/messages?key=AIzaSyDdI0hCZtE6vySjMm-WEfRq3CPzqKqqsHI&token=WhACImSKOnMpTU-3iyj92I-y6xmDd3KDHZtVY7GehNQ"

BASE_PATH = './'
MODEL_PATH = os.path.join(BASE_PATH, 'bigc_model.pth')
BKK_TZ = timezone(timedelta(hours=7))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Authentication (Sheet & Drive) ---
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive']
try:
    from google.oauth2.service_account import Credentials
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaFileUpload
    creds = Credentials.from_service_account_file('credentials.json', scopes=SCOPES)
    gc = gspread.authorize(creds)
    drive_service = build('drive', 'v3', credentials=creds)
except Exception as e:
    print(f"⚠️ [Colab Mode / Auth Issue]: {e}")

def get_bkk_now(): 
    return datetime.now(BKK_TZ)

def format_to_bkk(date_input):
    try:
        if isinstance(date_input, str) and date_input.replace('.', '', 1).isdigit():
            date_input = float(date_input)
        if isinstance(date_input, (int, float)):
            val = date_input if date_input < 1e11 else date_input / 1000.0
            dt = datetime.fromtimestamp(val, timezone.utc)
        else:
            clean_str = str(date_input).replace('Z', '+00:00').replace("'", "").strip()
            dt = datetime.fromisoformat(clean_str[:19] + '+00:00')
        return dt.astimezone(BKK_TZ).strftime('%Y-%m-%d %H:%M:%S')
    except: 
        return str(date_input).replace("'", "").strip()

def sanitize_for_sheets(text):
    if not text: return "-"
    clean_text = re.sub(r'[\x00-\x1F\x7F-\x9F]', '', str(text))
    clean_text = clean_text.replace('\n', ' ').strip()
    if clean_text.startswith(('=', '+', '-', '@')):
        clean_text = f"'{clean_text}"
    return clean_text[:2500]

def send_google_chat_message(message, webhook_url):
    if not webhook_url or not webhook_url.startswith("http"): return
    try:
        res = requests.post(webhook_url, headers={"Content-Type": "application/json"}, json={"text": message})
        if res.status_code == 200: 
            print("✅ ส่ง Google Chat สำเร็จ!")
    except Exception as e: 
        print(f"⚠️ Google Chat Send Exception: {e}")

def generate_summary_message(name_group, p_list, raw, unique, dup, chat):
    if not raw and not unique: return None 
    msg = f"📊 อัปเดตข้อมูลจาก Logo Hunter ({name_group})\n🔗 แพลตฟอร์ม: {', '.join(p_list)}\n📥 ดึงมาทั้งหมด: {len(raw)} โพสต์\n✨ ข้อมูลใหม่: {len(unique)} โพสต์\n"
    if chat:
        msg += f"🎯 พบโลโก้: {len(chat)} โพสต์\n\n🚨 ลิงก์โพสต์ที่พบโลโก้:\n"
        for item in chat[:15]: 
            msg += f"• {item['url']}\n  (🖼️ รูปหลักฐาน: {item['img_url']})\n"
        if len(chat) > 15: 
            msg += f"• ... และอื่นๆ อีก {len(chat) - 15} รายการ\n"
    elif unique:
        msg += f"🎯 พบโลโก้: 0 โพสต์\n\n📌 ลิงก์โพสต์ใหม่ (ส่งตรวจสอบแล้ว):\n"
        for u in unique[:15]: 
            msg += f"• {u['url']}\n"
        if len(unique) > 15: 
            msg += f"• ... และอื่นๆ อีก {len(unique) - 15} รายการ\n"
    msg += "\n✅ ตรวจสอบรายละเอียดได้ใน Sheet Log"
    return msg

def update_heartbeat(ws_control):
    try: 
        ws_control.update_cell(9, 2, get_bkk_now().strftime('%Y-%m-%d %H:%M:%S'))
    except: 
        pass

def load_ai_model():
    if os.path.exists(MODEL_PATH):
        try:
            m = torch.load(MODEL_PATH, map_location=device, weights_only=False)
            if hasattr(m, 'eval'): m.eval()
            print("🧠 AI Model Engine: ONLINE")
            return m
        except: 
            pass
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
        if pred.item() == 0 and conf.item() > 0.85: 
            return True, conf.item(), Image.fromarray(rgb)
    except: 
        pass
    return False, 0.0, None

def save_evidence(image_pil, video_id, timestamp_str):
    try:
        safe_ts = str(timestamp_str).replace(":", "_")
        local_filename = f"DETECT_{video_id}_{safe_ts}.jpg"
        image_pil.save(local_filename)
        
        if GDRIVE_FOLDER_ID and 'drive_service' in globals():
            file_metadata = {'name': local_filename, 'parents': [GDRIVE_FOLDER_ID]}
            media = MediaFileUpload(local_filename, mimetype='image/jpeg', resumable=True)
            file = drive_service.files().create(body=file_metadata, media_body=media, fields='id, webViewLink').execute()
            drive_service.permissions().create(fileId=file.get('id'), body={'type': 'anyone', 'role': 'reader'}).execute()
            url = file.get('webViewLink')
            os.remove(local_filename)
            return f'=HYPERLINK("{url}", "🖼️ กดดูรูปภาพ")', url
            
        file_path = os.path.join(RESULTS_PATH, local_filename)
        image_pil.save(file_path)
        drive_link = f"https://drive.google.com/drive/search?q={urllib.parse.quote(local_filename)}"
        return f'=HYPERLINK("{drive_link}", "🖼️ กดดูรูปภาพ")', drive_link
    except Exception as e:
        print(f" ⚠️ Upload Error: {e}")
        return "-", "-"

def fetch_data(platforms, keywords, max_res, days_back):
    all_videos = []
    cutoff_utc = datetime.now(timezone.utc) - timedelta(days=days_back)
    
    for plat in platforms:
        for kw in keywords:
            print(f"📡 Searching '{kw}' on {plat}...")
            
            if plat == 'YouTube':
                success = False
                for yt_key in YOUTUBE_API_KEYS:
                    if not yt_key or "ใส่_" in yt_key: continue
                    try:
                        params = {'part': 'snippet', 'q': kw, 'key': yt_key, 'maxResults': max_res, 'type': 'video', 'publishedAfter': cutoff_utc.isoformat().replace('+00:00', 'Z')}
                        res = requests.get("https://www.googleapis.com/youtube/v3/search", params=params).json()
                        
                        if 'error' in res:
                            print(f"  ⚠️ YouTube Key (ลงท้าย {yt_key[-4:]}) โควตาเต็มหรือมีปัญหา กำลังสลับ Key ถัดไป...")
                            continue
                            
                        for item in res.get('items', []):
                            all_videos.append({
                                'url': f"https://www.youtube.com/watch?v={item['id']['videoId']}", 
                                'title': item['snippet']['title'], 
                                'platform': 'YouTube', 
                                'user': item['snippet']['channelTitle'], 
                                'date': format_to_bkk(item['snippet']['publishedAt']), 
                                'image_url': item['snippet']['thumbnails']['high']['url'], 
                                'id': item['id']['videoId'],
                                'is_old': False # YouTube ดึงเฉพาะของใหม่มาให้อยู่แล้ว
                            })
                        
                        success = True
                        break 
                    except Exception as e:
                        print(f"  ⚠️ YouTube Request Error (Key ...{yt_key[-4:]}): {e}")
                        continue
                
                if not success:
                    print(f"  ❌ YouTube API Keys ทั้งหมดโควตาเต็ม หรือไม่ได้ตั้งค่า!")

            else:
                actor = {'TikTok': 'clockworks/tiktok-scraper', 'Instagram': 'apify/instagram-hashtag-scraper', 'Facebook': 'apify/facebook-search-scraper'}.get(plat)
                if not actor: continue
                inp = {"resultsLimit": max_res, "maxItems": max_res, "resultsPerPage": max_res, "proxyConfiguration": {"useApifyProxy": True}}
                if plat in ['TikTok', 'Instagram']: 
                    inp["hashtags"] = [kw.replace(" ","")]
                else: 
                    inp["searchTerms"] = kw
                
                success = False
                for apify_token in APIFY_TOKENS:
                    if not apify_token or "ใส่_" in apify_token: continue
                    try:
                        client = ApifyClient(apify_token)
                        run = client.actor(actor).call(run_input=inp, timeout_secs=120)
                        
                        for item in client.dataset(run["defaultDatasetId"]).list_items().items:
                            if not item or 'error' in item: continue

                            # 🛑 คำนวณวันที่: ติดป้ายว่า 'เก่า' แต่ไม่เตะทิ้งจาก Array 
                            raw_date = item.get('createTime') or item.get('timestamp') or item.get('date') or item.get('create_time')
                            is_old = False
                            try:
                                if raw_date:
                                    rd = float(raw_date) if isinstance(raw_date, str) and raw_date.replace('.', '', 1).isdigit() else raw_date
                                    if isinstance(rd, (int, float)):
                                        dt_utc = datetime.fromtimestamp(rd if rd < 1e11 else rd/1000.0, timezone.utc)
                                    else:
                                        clean_str = str(rd).replace('Z', '+00:00')[:19] + '+00:00'
                                        dt_utc = datetime.fromisoformat(clean_str)
                                    
                                    if dt_utc < cutoff_utc:
                                        is_old = True
                            except: 
                                pass
                            
                            user = "Unknown"
                            if isinstance(item.get('authorMeta'), dict): user = item['authorMeta'].get('name') or item['authorMeta'].get('nickName') or "Unknown"
                            elif isinstance(item.get('author'), dict): user = item['author'].get('uniqueId') or item['author'].get('nickname') or "Unknown"
                            elif isinstance(item.get('author'), str): user = item['author']
                            if user == "Unknown" or not user: user = item.get('authorNickname') or item.get('authorName') or item.get('ownerUsername') or item.get('author_id') or "Unknown"
                                
                            title_raw = item.get('caption') or item.get('text') or item.get('desc') or item.get('title') or item.get('video_description') or "No Title"
                            if isinstance(title_raw, dict): title_raw = title_raw.get('text', 'No Title')
                            title = str(title_raw)
                            
                            item_id = item.get('id') or item.get('video', {}).get('id') or item.get('video_id')
                            if not item_id: item_id = str(random.randint(10000, 99999))
                            
                            v_url = item.get('webVideoUrl') or item.get('videoWebUrl') or item.get('url') or item.get('postUrl') or item.get('video_url')
                            if not v_url and plat == 'TikTok': v_url = f"https://www.tiktok.com/@{user}/video/{item_id}"
                            
                            image_url = item.get('displayUrl') or item.get('imageUrl') or item.get('coverUrl') or item.get('video', {}).get('cover') or item.get('origin_cover')

                            if v_url:
                                all_videos.append({
                                    'url': str(v_url), 'title': str(title)[:200], 'platform': str(plat), 'user': str(user), 
                                    'date': format_to_bkk(raw_date), 'id': str(item_id), 'image_url': str(image_url) if image_url else None,
                                    'is_old': is_old # แปะป้ายบอกว่าคลิปนี้เก่า (is_old: True)
                                })
                        
                        success = True
                        break 
                    except Exception as e:
                        print(f"  ⚠️ Apify Token (ลงท้าย ...{apify_token[-4:]}) เครดิตหมด กำลังสลับ Token ถัดไป...")
                        continue

                if not success:
                    print(f"  ❌ Apify Tokens ทั้งหมดเครดิตหมด หรือไม่ได้ตั้งค่าสำหรับ {plat}!")
                    
    return all_videos

def main():
    if not SHEET_ID and 'auth' in globals():
        auth.authenticate_user()
        creds, _ = default()
        gc = gspread.authorize(creds)
        target_sheet_id = "1RwsclzY3ssqnXSccdsclk53Lg6CJUvhKy9TMRXXgp6k" 
    else:
        target_sheet_id = SHEET_ID

    sh = gc.open_by_key(target_sheet_id)
    ws_data = sh.worksheet("Apify")
    ws_control = sh.worksheet("Control_Panel")
    try: 
        ws_logs = sh.worksheet("Scan_Logs")
    except:
        ws_logs = sh.add_worksheet(title="Scan_Logs", rows=1000, cols=4)
        ws_logs.append_row(['Timestamp', 'ลิสต์รายการที่ซ้ำ', 'ลิสต์รายการที่ไม่ซ้ำ', 'Platforms'])

    engine = load_ai_model()
    scanned_memory = set()
    print(f"\n🚀 LOGO HUNTER v7.2 ONLINE")

    while True:
        try:
            update_heartbeat(ws_control)
            config = ws_control.col_values(2)
            status = str(config[0]).strip() if len(config) > 0 else "🔴 Stop"

            if 'Start' not in status and '🟢' not in status:
                print(f"\r⏳ [Standby] Waiting for command... ({get_bkk_now().strftime('%H:%M:%S')})", end="")
                time.sleep(15)
                continue

            platforms = [p.strip() for p in config[1].split(',')]
            keywords = [k.strip() for k in config[2].split(',')]
            
            # 🧠 กลับมาใช้ Time Filter รูปแบบเดิม (1=Day, 2=Week, 3=Month)
            days_back = 1 if '1' in str(config[3]) else 7 if '2' in str(config[3]) else 30
            
            max_res = int(config[6]) if str(config[6]).isdigit() else 5
            run_mode = str(config[4]).strip()

            print(f"\n\n🔥 Processing Batch: {get_bkk_now().strftime('%H:%M:%S')}")
            raw_list = fetch_data(platforms, keywords, max_res, days_back)
            processed_urls = set(ws_data.col_values(7)[1:])

            unique_list, duplicate_list, seen_in_run = [], [], set()
            
            # 📂 แยกประเภทข้อมูล (ใหม่สุดๆ ไปขวา / ซ้ำและเก่า ไปซ้าย)
            for v in raw_list:
                # ถ้าเคยตรวจแล้ว, อยู่ใน memory, หรือ "เป็นคลิปเก่า(is_old)" ให้จับยัดลงซ้าย
                if v['url'] in processed_urls or v['url'] in scanned_memory or v['url'] in seen_in_run or v.get('is_old', False): 
                    duplicate_list.append(v)
                else: 
                    seen_in_run.add(v['url'])
                    unique_list.append(v)

            print(f"📋 Found: {len(raw_list)} (New: {len(unique_list)} / Dup & Old: {len(duplicate_list)})")
            
            # 📝 บันทึก Log (ของใหม่และของซ้ำแยกซ้ายขวาตามปกติ)
            new_text = "\n".join([f"[{u['platform']}] {sanitize_for_sheets(u['title'])[:50]}... -> {u['url']}" for u in unique_list])
            dup_text = "\n".join([f"[{d['platform']}] {sanitize_for_sheets(d['title'])[:50]}... -> {d['url']}" for d in duplicate_list])

            try: 
                ws_logs.insert_row([get_bkk_now().strftime('%Y-%m-%d %H:%M:%S'), str(dup_text) if dup_text else "-", str(new_text) if new_text else "-", str(", ".join(platforms))], index=2)
                time.sleep(2)
            except: 
                pass

            chat_summary = []
            batch_rows_to_insert = []

            # 👁️ สแกนเฉพาะคลิปที่ใหม่เท่านั้น (ข้ามคลิปซ้ำและคลิปเก่าทั้งหมด)
            for v in unique_list:
                print(f"👁️ Scanning: [{v['user']}] - {v['title'][:40]}...")
                found, final_ts, best_img = False, "-", None
                
                if engine is not None:
                    try:
                        with yt_dlp.YoutubeDL({'format': 'best', 'quiet': True, 'nocheckcertificate': True, 'socket_timeout': 10}) as ydl:
                            stream = ydl.extract_info(v['url'], download=False).get('url')
                            if stream:
                                cap, fps = cv2.VideoCapture(stream), 30
                                for i in range(150):
                                    ret, frame = cap.read()
                                    if not ret: break
                                    if i % 30 == 0:
                                        hit, sc, img = predict_logo(engine, frame)
                                        if hit: 
                                            found, best_img, final_ts = True, img, f"{int((i//fps)//60):02d}:{int((i//fps)%60):02d}"
                                            break
                                cap.release()
                    except: 
                        pass

                    if not found and v.get('image_url'):
                        try:
                            img = cv2.imdecode(np.asarray(bytearray(requests.get(v['image_url'], timeout=10).content), dtype=np.uint8), cv2.IMREAD_COLOR)
                            hit, sc, img_res = predict_logo(engine, img)
                            if hit: 
                                found, best_img, final_ts = True, img_res, "Image"
                        except: 
                            pass

                clean_title = sanitize_for_sheets(v.get('title', 'No Title'))
                clean_user = sanitize_for_sheets(v.get('user', 'Unknown'))
                clean_platform = sanitize_for_sheets(v.get('platform', '-'))
                clean_date = sanitize_for_sheets(v.get('date', '-'))
                clean_url = sanitize_for_sheets(v.get('url', '-'))

                if found:
                    print(f"  🎯 Logo Found at {final_ts}!")
                    formula, raw_drive_link = save_evidence(best_img, v['id'], final_ts)
                    detect_status = "Yes"
                    chat_summary.append({'url': clean_url, 'img_url': str(raw_drive_link), 'platform': clean_platform})
                else: 
                    print("  ❌ No logo.")
                    formula, raw_drive_link = "-", "-"
                    detect_status = "No"
                    
                # 🌟 เก็บเวลาปัจจุบันที่บอททำงาน (Timestamp)
                current_run_time = get_bkk_now().strftime('%Y-%m-%d %H:%M:%S')

                # 🔧 นำข้อมูลแถวนี้เก็บลงตะกร้า (แทรก '-' เพื่อดัน Timestamp ไปอยู่คอลัมน์ J)
                row_data = [
                    clean_date, clean_title, clean_platform,
                    clean_user, str(detect_status), str(final_ts),
                    clean_url, str(formula), "-", current_run_time
                ]
                batch_rows_to_insert.append(row_data)
                scanned_memory.add(v['url'])

            if batch_rows_to_insert:
                try:
                    batch_rows_to_insert.reverse() 
                    ws_data.insert_rows(batch_rows_to_insert, row=2, value_input_option='USER_ENTERED')
                    print("  ✅ บันทึกลง Sheet 'Apify' สำเร็จทั้งหมด!")
                except Exception as sheet_err:
                    print(f"  ❌ เขียนลง Sheet 'Apify' ไม่สำเร็จ: {sheet_err}")

            if ENABLE_STATUS_MESSAGE or len(chat_summary) > 0:
                msg_all = generate_summary_message("รวมทุกแพลตฟอร์ม", platforms, raw_list, unique_list, duplicate_list, chat_summary)
                if msg_all: send_google_chat_message(msg_all, GOOGLE_CHAT_WEBHOOK)

            tk_raw = [x for x in raw_list if x['platform'] == 'TikTok']
            tk_uni = [x for x in unique_list if x['platform'] == 'TikTok']
            tk_dup = [x for x in duplicate_list if x['platform'] == 'TikTok']
            tk_chat = [x for x in chat_summary if x['platform'] == 'TikTok']
            tk_plats = ['TikTok'] if 'TikTok' in platforms else []

            if tk_plats and (ENABLE_STATUS_MESSAGE or len(tk_chat) > 0):
                msg_tk = generate_summary_message("TikTok", tk_plats, tk_raw, tk_uni, tk_dup, tk_chat)
                if msg_tk: send_google_chat_message(msg_tk, GOOGLE_CHAT_WEBHOOK_TIKTOK)

            if not SHEET_ID and 'auth' in globals():
                if 'Run Once' in run_mode:
                    print("\n🏁 Run Once finished. Returning to Standby.")
                    ws_control.update_cell(1, 2, '🔴 Stop')
                else:
                    interval = int(config[5]) if str(config[5]).isdigit() else 60
                    print(f"\n💤 Next iteration in {interval} mins...")
                    for _ in range(interval * 6):
                        update_heartbeat(ws_control)
                        time.sleep(10)
            else:
                break 

        except Exception as e:
            print(f"\n❌ Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    main()
