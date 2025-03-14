# Smart Parking System using YOLOv8 & OpenCV  

## Project Overview  
With the increasing number of vehicles in urban areas, finding a parking space has become a significant challenge, leading to traffic congestion and fuel wastage. This **Smart Parking System** leverages **YOLOv8 and OpenCV** to detect vehicles and monitor parking slots in real-time using CCTV cameras.  

## Features  
**Real-time Parking Slot Detection** – Uses **YOLOv8** to classify parking slots as occupied (Red) or available (Green).  
**Shortest Path Navigation** – Implements the **Euclidean algorithm** to direct drivers to the nearest available parking slot.  


## Technologies Used  
- **YOLOv8** – Object detection for identifying vehicles and parking slots  
- **OpenCV** – Image processing for real-time monitoring  
- **Python** – Backend processing  
- **CCTV Cameras** – Capturing live parking footage  
- **Euclidean Algorithm** – Finding the shortest path to an available slot  

## How It Works  
1. Captures real-time footage from CCTV cameras.  
2. Uses **YOLOv8** to detect vehicles and classify parking slots as occupied or available.  
3. Computes the **shortest route** to the nearest slot using the **Euclidean algorithm**.  
4. Displays **slot availability and navigation path** to the driver.  


