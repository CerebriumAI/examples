import { useState, useEffect, useCallback, useRef } from "react";
import AgoraRTC from "agora-rtc-sdk-ng";

const client = AgoraRTC.createClient({ mode: "rtc", codec: "vp8" });

export default function VideoCall({ onAgePrediction }) {
  const [localVideoTrack, setLocalVideoTrack] = useState(null);
  const [remoteUsers, setRemoteUsers] = useState([]);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [socketReady, setSocketReady] = useState(false);
  const socketRef = useRef(null);
  const reconnectTimeoutRef = useRef(null);
  const isConnectingRef = useRef(false);

  // Use environment variables with proper access for Create React App
  const AGORA_APP_ID = import.meta.env.VITE_AGORA_APP_ID;
  const CHANNEL = import.meta.env.VITE_AGORA_CHANNEL;
  const TOKEN = import.meta.env.VITE_AGORA_TOKEN;

  // Use environment variable for WebSocket endpoint
  const WEBSOCKET_ENDPOINT = import.meta.env.VITE_WEBSOCKET_ENDPOINT;

  // Function to establish WebSocket connection - only if not already connecting or connected
  const connectWebSocket = useCallback(() => {
    // Don't create a new connection if we're already connecting or have a valid connection
    if (isConnectingRef.current) {
      console.log(
        "Already attempting to connect, skipping duplicate connection attempt",
      );
      return;
    }

    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      console.log("WebSocket already connected, skipping connection attempt");
      setSocketReady(true);
      return;
    }

    // Clear any pending reconnect timeouts
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    console.log("Attempting to connect WebSocket...");
    isConnectingRef.current = true;

    // Close existing socket if in a bad state
    if (
      socketRef.current &&
      socketRef.current.readyState !== WebSocket.CLOSED
    ) {
      console.log("Closing existing socket before creating new one");
      socketRef.current.close();
    }

    try {
      // Initialize WebSocket connection with token as query parameter
      const wsUrl = new URL(WEBSOCKET_ENDPOINT);
      const ws = new WebSocket(wsUrl.toString());

      ws.onopen = () => {
        console.log("WebSocket connection established");
        setSocketReady(true);
        isConnectingRef.current = false;
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log("Received message:", data.predicted_age);
          setAnalysisResult(data);

          if (data) {
            console.log("Received message from server:", data.predicted_age);
            const age = data.predicted_age;
            if (age) {
              onAgePrediction(age);
            }
          }
        } catch (error) {
          console.error("Error parsing WebSocket message:", error);
        }
      };

      ws.onerror = (error) => {
        console.error("WebSocket error:", error);
        isConnectingRef.current = false;
        setSocketReady(false);
      };

      ws.onclose = (event) => {
        console.log("WebSocket connection closed:", event.code, event.reason);
        isConnectingRef.current = false;
        setSocketReady(false);

        // Only attempt to reconnect if component is still mounted
        // and we don't already have a reconnection scheduled
        if (!reconnectTimeoutRef.current) {
          reconnectTimeoutRef.current = setTimeout(() => {
            reconnectTimeoutRef.current = null;
            console.log("Attempting to reconnect...");
            connectWebSocket();
          }, 3000);
        }
      };

      socketRef.current = ws;
    } catch (error) {
      console.error("Error creating WebSocket:", error);
      isConnectingRef.current = false;
    }
  }, []);

  useEffect(() => {
    // Start the call when component mounts
    startCall();

    // Initialize WebSocket connection - only once on mount
    connectWebSocket();

    // Cleanup when component unmounts
    return () => {
      cleanup();

      // Clear any reconnection timeouts
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }

      // Close the socket if it exists
      if (socketRef.current) {
        // Prevent reconnection attempts when intentionally closing
        const socket = socketRef.current;
        socket.onclose = null;
        socket.close();
        socketRef.current = null;
      }
    };
  }, []);

  const startCall = async () => {
    try {
      console.log("Starting call...");
      // Join the channel
      await client.join(AGORA_APP_ID, CHANNEL, TOKEN, null);
      console.log("Joined channel successfully");

      // Create and publish local tracks with constraints
      const cameraConfig = {
        encoderConfig: "1080p",
        facingMode: "user",
      };

      console.log("Creating camera and microphone tracks...");
      const tracks = await AgoraRTC.createMicrophoneAndCameraTracks(
        undefined,
        cameraConfig,
      );
      const [audioTrack, videoTrack] = tracks;

      console.log("Tracks created:", videoTrack);

      if (!videoTrack) {
        throw new Error("Failed to create video track");
      }

      // Set local video track
      setLocalVideoTrack(videoTrack);

      // Play the local video in the UI first
      const localContainer = document.querySelector(".video-frame");
      if (localContainer) {
        console.log("Playing video in local container");
        videoTrack.play(localContainer);
      } else {
        console.error("Local container not found");
      }

      // Start frame capture
      console.log("Starting frame capture...");
      startFrameCapture(videoTrack);

      // Publish tracks
      console.log("Publishing tracks...");
      await client.publish([audioTrack, videoTrack]);
      console.log("Tracks published successfully");

      // Handle remote users
      client.on("user-published", handleUserPublished);
      client.on("user-unpublished", handleUserUnpublished);
    } catch (error) {
      console.error("Error starting call:", error);
    }
  };

  const startFrameCapture = async (videoTrack) => {
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d");

    const videoElement = document.querySelector(".video-frame");
    const overlayCanvas = document.querySelector(".overlay-canvas");
    const overlayContext = overlayCanvas.getContext("2d");

    if (!videoElement || !overlayCanvas) {
      console.error("Video or overlay elements not found");
      return;
    }

    // Wait for video to be playing and have dimensions
    await new Promise((resolve) => {
      const checkVideo = () => {
        if (videoElement.videoWidth && videoElement.videoHeight) {
          console.log(
            "Video dimensions:",
            videoElement.videoWidth,
            videoElement.videoHeight,
          );
          // Set overlay canvas dimensions to match video
          overlayCanvas.width = videoElement.videoWidth;
          overlayCanvas.height = videoElement.videoHeight;
          resolve();
        } else {
          setTimeout(checkVideo, 100);
        }
      };
      checkVideo();
    });

    // Capture frame every second
    const intervalId = setInterval(async () => {
      try {
        // Set canvas dimensions to match video
        canvas.width = videoElement.videoWidth;
        canvas.height = videoElement.videoHeight;

        // Draw video frame to canvas
        context.drawImage(videoElement, 0, 0);

        // Get base64 data URL from canvas
        const base64Image = canvas.toDataURL("image/jpeg", 0.8);
        const base64Data = base64Image.split(",")[1];

        // Debug: Check if we have frame data
        console.log("Captured frame data length:", base64Data?.length);

        // Only try to send if we have an open socket
        if (
          socketRef.current &&
          socketRef.current.readyState === WebSocket.OPEN
        ) {
          console.log("Sending frame through WebSocket");

          try {
            // Create the message payload
            const message = JSON.stringify({
              frame_data: base64Data,
              prompt: "detect faces",
              num_inference_steps: 8,
              guidance_scale: 1.0,
            });

            // Send the message
            socketRef.current.send(message);
            console.log("Frame sent successfully");
          } catch (sendError) {
            console.error("Error sending message:", sendError);

            // Only try to reconnect if we're not already in the process
            if (!isConnectingRef.current) {
              setSocketReady(false);
              connectWebSocket();
            }
          }
        } else {
          console.warn(
            "WebSocket not ready, skipping frame. State:",
            socketRef.current ? socketRef.current.readyState : "No socket",
          );

          // Only try to reconnect if we're not already in the process
          if (
            !isConnectingRef.current &&
            (!socketRef.current ||
              socketRef.current.readyState !== WebSocket.CONNECTING)
          ) {
            connectWebSocket();
          }
        }

        // Draw the analysis results on the overlay canvas
        if (analysisResult && analysisResult.predictions) {
          // Clear previous overlay
          overlayContext.clearRect(
            0,
            0,
            overlayCanvas.width,
            overlayCanvas.height,
          );

          overlayContext.strokeStyle = "#00ff00";
          overlayContext.lineWidth = 2;
          overlayContext.font = "16px Arial";
          overlayContext.fillStyle = "#00ff00";

          analysisResult.predictions.forEach((pred) => {
            // Assuming the API returns bounding box coordinates
            const { x, y, width, height } = pred.bbox;

            // Draw bounding box
            overlayContext.strokeRect(x, y, width, height);

            // Draw label if available
            if (pred.label) {
              overlayContext.fillText(pred.label, x, y - 5);
            }
          });
        }
      } catch (error) {
        console.error("Error capturing frame:", error);
      }
    }, 1000);

    // Store the interval ID for cleanup
    return intervalId;
  };

  const handleUserPublished = async (user, mediaType) => {
    await client.subscribe(user, mediaType);
    if (mediaType === "video") {
      setRemoteUsers((prev) => [...prev, user]);
    }
    if (mediaType === "audio") {
      user.audioTrack?.play();
    }
  };

  const handleUserUnpublished = (user) => {
    setRemoteUsers((prev) => prev.filter((u) => u.uid !== user.uid));
  };

  const cleanup = async () => {
    // Stop all tracks and leave channel
    localVideoTrack?.stop();
    localVideoTrack?.close();
    await client.leave();

    // Clear state
    setLocalVideoTrack(null);
    setRemoteUsers([]);
  };

  // Add a debug effect to monitor socket state
  useEffect(() => {
    console.log(
      "Socket state updated:",
      socketRef.current ? "Socket exists" : "No socket",
      "Ready:",
      socketReady,
    );

    // If we have a socket but it's not marked ready, check its state
    if (
      socketRef.current &&
      !socketReady &&
      socketRef.current.readyState === WebSocket.OPEN
    ) {
      console.log("Socket is open but not marked ready, fixing...");
      setSocketReady(true);
    }
  }, [socketRef, socketReady]);

  return (
    <div className="video-container">
      {/* Local video */}
      <div className="video-player">
        <div
          style={{
            position: "relative",
            width: "600px", // Increased from 900px
            height: "400px", // Increased from 675px
            margin: "0 auto",
          }}
        >
          <video
            className="video-frame"
            style={{
              width: "100%",
              height: "100%",
              objectFit: "contain", // This prevents stretching
              border: "1px solid #ccc",
              backgroundColor: "#f0f0f0",
            }}
            autoPlay
            playsInline
            muted
            ref={(el) => {
              if (el && localVideoTrack) {
                console.log("Playing local video track");
                localVideoTrack.play(el);
              }
            }}
          ></video>
          <canvas
            className="overlay-canvas"
            style={{
              position: "absolute",
              top: 0,
              left: 0,
              width: "100%",
              height: "100%",
              pointerEvents: "none",
            }}
          ></canvas>
        </div>
      </div>

      {/* Remote videos */}
      {remoteUsers.map((user) => (
        <div key={user.uid} className="video-player">
          <div
            style={{
              position: "relative",
              width: "600px", // Increased from 900px
              height: "400px", // Increased from 675px
              margin: "0 auto",
            }}
          >
            <video
              className="video-frame"
              style={{
                width: "100%",
                height: "100%",
                objectFit: "contain",
                border: "1px solid #ccc",
                backgroundColor: "#f0f0f0",
              }}
              autoPlay
              playsInline
              ref={(el) => {
                if (el && user.videoTrack) {
                  user.videoTrack.play(el);
                }
              }}
            ></video>
          </div>
          <p>Remote User {user.uid}</p>
        </div>
      ))}
    </div>
  );
}
