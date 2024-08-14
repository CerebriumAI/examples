from daily import *
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import io
import queue
import threading
import time
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
import torch.nn.functional as F
import libsql_experimental as libsql
from supabase import create_client
from serpapi import GoogleSearch
from cerebrium import get_secret
import torch

model = YOLO("./best.pt")  # load a pretrained model (recommended for training)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# uncomment to use Turso
# processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
# vision_model = AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True)

# url = get_secret("TURSO_DB_URL")
# auth_token = get_secret("TURSO_AUTH_TOKEN")

# conn = libsql.connect("products.db", sync_url=url, auth_token=auth_token)
# conn.sync()

supabase = create_client(get_secret("SUPABASE_URL"), get_secret("SUPABASE_ANON"))

class ObjectDetection(EventHandler):
  def __init__(self, room_url):
    self.client = CallClient(event_handler = self)
    self.is_running = True
    self.message_sent = False
    self.queue = queue.Queue()
    self.room_url = room_url
    self.consecutive_detections = {}  # To track consecutive detections


    self.camera = Daily.create_camera_device("my-camera", width = 1280, height = 720, color_format = "RGB")
    self.client.update_inputs({
        "camera": {
            "isEnabled": True,
            "settings": {
            "deviceId": "my-camera"
            }
        },
        "microphone": False
    })

    #Since frames are sent every 30ms, we only want to send one every 1.35s
    self.frame_cadence = 5
    self.frame_count = 0
    self.thread_count = 0
    self.detected_items = set()  # Set to keep track of detected items


  def on_participant_left(self, participant, reason):
    if len(self.client.participant_counts()) <=2: ##count is before the user has left
      self.is_running = False

  def on_participant_joined(self, participant):
    if not participant["info"]['isLocal']:
      self.client.set_video_renderer(participant["id"], callback = self.on_video_frame)

  def on_video_frame(self, participant, frame):
    self.frame_count += 1
    if self.frame_count >= self.frame_cadence and self.thread_count < 5:
      self.frame_count = 0
      self.thread_count += 1

      self.queue.put({"buffer": frame.buffer, "width": frame.width, "height": frame.height})

      worker_thread = threading.Thread(target=self.process_frame, daemon=True)
      worker_thread.start()
    
  def process_frame(self):
    item = self.queue.get()

    try:
        image = Image.frombytes('RGBA', (item["width"], item["height"]), item["buffer"])
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGBA2BGR)

        detections = model.predict(image, verbose=False)
        
        if len(detections[0].boxes) > 0:
            for box in detections[0].boxes:
                class_id = int(box.cls.item())
                class_name = model.names[class_id]
                confidence = float(box.conf.item())

                if class_name.lower() == 'person':
                    continue
                
                print(f"Detected {class_name} with {confidence:.2%} confidence")
                if confidence > 0.55:
                    if class_name not in self.consecutive_detections:
                        self.consecutive_detections[class_name] = 1
                    else:
                        self.consecutive_detections[class_name] += 1
                else:
                    self.consecutive_detections[class_name] = 0
                
                if self.consecutive_detections.get(class_name, 0) > 3:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    
                    
                    if class_name not in self.detected_items and class_name.lower() != 'person':
                        detected_object = image[y1:y2, x1:x2]
                        self.detected_items.add(class_name)
                        self.search_image(detected_object)
                
                
                # Reset counts for classes not detected in this frame
                for class_name in list(self.consecutive_detections.keys()):
                    if class_name not in [model.names[int(box.cls.item())] for box in detections[0].boxes]:
                        self.consecutive_detections[class_name] = 0

    except Exception as e:
        print(f'\nIssue converting image and detecting: {e}')
    
    self.thread_count -= 1
    self.queue.task_done()
    return

  def join(self, url):
     self.client.join(url)
     time.sleep(4)

  def isRunning(self):
    return self.is_running
  
  def search(self,url):
    params = {
        "engine": "google_lens",
        "url": url,
        "api_key": f"{get_secret("SERP_API")}"
    }

    search = GoogleSearch(params)
    results = search.get_dict()
    amazon_results = [item for item in results["visual_matches"] if item.get("source") == "Amazon.com" and "price" in item]
    
    if amazon_results:
        # Return the first Amazon result with a price
        return amazon_results[0]
    else:
        # If no Amazon results, find the first result with a price
        for item in results["visual_matches"]:
            if "price" in item:
                return item
    
    
    # If no results with price found
    return None

  def search_db(self, image):
    inputs = processor(image, return_tensors="pt")
    img_emb = vision_model(**inputs).last_hidden_state
    img_embeddings = (F.normalize(img_emb[:, 0], p=2, dim=1)).detach().numpy().tolist()

    results = conn.execute("SELECT * FROM products WHERE vector_distance_cos(embedding, ?) LIMIT 1;", (str(img_embeddings[0]),)).fetchall()
    return results

  def uploadImage(self, image):
    import uuid
    import tempfile
    import os
    # Generate a random UUID
    random_uuid = uuid.uuid4()
    
    # Convert the UUID to a string and use it in the filename
    filename = f"{self.room_url}_{random_uuid}.png"
    with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
        temp_filename = temp_file.name
        cv2.imwrite(temp_filename, image)
    
    # Upload the file
    try:
        with open(temp_filename, 'rb') as f:
            result = supabase.storage.from_("products").upload(
                file=f,
                path=filename,
                file_options={"content-type": "image/png"}
            )
        public_url = supabase.storage.from_("products").get_public_url(filename)
        
        return {"result": result, "url": public_url}
    finally:
        # Clean up the temporary file
        os.unlink(temp_filename)
  
  def search_image(self, image):
    
    #if using Turso uncomment below
    # self.search_db(image)

    #add item to supabase
    image_result = self.uploadImage(image)
    search_result = self.search(image_result['url'])

    supabase.table("products").insert({
       "name": search_result['title'],
       "description": "",
       "price": str(search_result.get('price', {}).get('extracted_value', 0.00)),
       "image_url": search_result["thumbnail"],
       "url": search_result['link'],
       "run_id": self.room_url,
       "original_image": image_result['url']
    }).execute()