import React, { useState, useEffect, useRef } from "react";
import Webcam from "react-webcam";
import * as faceapi from "face-api.js";
import axios from "axios";

const BASE_URL = import.meta.env.VITE_BACKEND_URL;

const PHOTOS_PER_POSITION = 10;
const POSITIONS = ["front", "right", "left"];
const TOTAL_PHOTOS_REQUIRED = PHOTOS_PER_POSITION * POSITIONS.length;

// Adjusted thresholds for greater tolerance
const LIGHTING_THRESHOLD_POOR = 25;
const LIGHTING_THRESHOLD_MODERATE = 45;
// Lower threshold for detection confidence to be more permissive
const DETECTION_THRESHOLD_POOR = 0.3;
const DETECTION_THRESHOLD_MODERATE = 0.5;
// Orientation thresholds
const ORIENTATION_THRESHOLD = {
  NOSE_DRIFT_SIDE: 0.06,  // Lower threshold makes side detection more sensitive
};

const CompleteRegistration = ({ rollNumber, email, onSubmit }) => {
  const [fullName, setFullName] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [branch, setBranch] = useState("");
  const [semester, setSemester] = useState("");
  const [batch, setBatch] = useState("");
  const [selectedSections, setSelectedSections] = useState([]);
  const [availableSections, setAvailableSections] = useState([]);
  const [capturedImages, setCapturedImages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [cameraActive, setCameraActive] = useState(false);
  const [currentPosition, setCurrentPosition] = useState(0);
  const [instructions, setInstructions] = useState("");
  const [boundingBoxColor, setBoundingBoxColor] = useState("#4caf50");
  const [faceDetected, setFaceDetected] = useState(false);
  const [detectionConfidence, setDetectionConfidence] = useState(0);
  const [lightingLevel, setLightingLevel] = useState(0);
  const [boundingBox, setBoundingBox] = useState(null);
  const [captureTimer, setCaptureTimer] = useState(null);
  const [debugMode, setDebugMode] = useState(true);
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [modelLoadingError, setModelLoadingError] = useState(false);
  const [currentOrientation, setCurrentOrientation] = useState("unknown");
  const [orientationScores, setOrientationScores] = useState({});
  const [captureCountdown, setCaptureCountdown] = useState(null);
  const [consecutiveCorrectOrientations, setConsecutiveCorrectOrientations] = useState(0);
    const [faceRegistrationComplete, setFaceRegistrationComplete] = useState(false); // New State
  const webcamRef = useRef(null);
  const canvasRef = useRef(null);

  const branchOptions = ["CSE", "ECE"];
  const semesterOptions = ["1", "2", "3", "4", "5", "6", "7", "8"];
  const batchOptions = rollNumber ? [`20${rollNumber.substring(0, 2)}`] : [];

  useEffect(() => {
    const loadModels = async () => {
      try {
        const MODEL_PATH = '/models'; // Relative path to models folder
        console.log("Loading models from:", MODEL_PATH);

        await Promise.all([
          faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_PATH),
          faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_PATH),
          faceapi.nets.faceRecognitionNet.loadFromUri(MODEL_PATH),
        ]);

        console.log("Face detection models loaded successfully");
        setModelsLoaded(true);
        setModelLoadingError(false);
      } catch (error) {
        console.error("Error loading face detection models:", error);
        setModelLoadingError(true);
        setModelsLoaded(false);
      }
    };

    // Only load models if they haven't been loaded yet
    if (!faceapi.nets.tinyFaceDetector.params) {
      loadModels();
    } else {
      setModelsLoaded(true);
    }
  }, []);

  useEffect(() => {
    if (branch && semester) {
      let sections = [];

      if (branch === "CSE" && semester === "6") {
        sections = [
          {
            category: "Main Sections (Choose One)",
            sections: ["CS31", "CS32"],
            type: "main"
          },
          {
            category: "Other Sections (Choose One)",
            sections: ["SC31", "SC32", "SC33"],
            type: "other"
          },
          {
            category: "HS Sections (Choose One)",
            sections: ["HS307", "HS308"],
            type: "hs"
          },
          {
            category: "Elective Sections (Choose One)",
            sections: ["CS300", "CS481", "CS653"],
            type: "elective"
          }
        ];
      } else if (branch === "ECE" && semester === "6") {
        sections = [
          {
            category: "Main Section (select this)",
            sections: ["EC3"],
            type: "main"
          },
          {
            category: "Lab Sections (Choose One)",
            sections: ["EC31", "EC32"],
            type: "lab"
          },
          {
            category: "Other Sections (Choose One)",
            sections: ["SC31", "SC32", "SC33"],
            type: "other"
          },
          {
            category: "HS Sections (Choose One)",
            sections: ["HS307", "HS308"],
            type: "hs"
          },
          {
            category: "Elective Sections (Choose One)",
            sections: ["EC300", "None"],
            type: "elective"
          }
        ];
      }

      setAvailableSections(sections);
    } else {
      setAvailableSections([]);
    }
  }, [branch, semester]);

  // Draw canvas over webcam for bounding box and debug info
  useEffect(() => {
    if (webcamRef.current && canvasRef.current && cameraActive) {
      const video = webcamRef.current.video;
      const canvas = canvasRef.current;

      if (video && video.readyState === 4) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (boundingBox) {
        ctx.strokeStyle = boundingBoxColor;
        ctx.lineWidth = 3;
        ctx.strokeRect(
          boundingBox.x,
          boundingBox.y,
          boundingBox.width,
          boundingBox.height
        );

        if (debugMode) {
          // Create a semi-transparent background for debug info
          ctx.fillStyle = "rgba(255, 255, 255, 0.8)";
          ctx.fillRect(10, 10, 350, 120);
          ctx.fillStyle = "#000000";
          ctx.font = "16px Arial";
          ctx.fillText(`Detection Confidence: ${detectionConfidence.toFixed(3)}`, 20, 30);
          ctx.fillText(`Lighting Level: ${lightingLevel.toFixed(1)}`, 20, 50);
          ctx.fillText(`Current Orientation: ${currentOrientation}`, 20, 70);
          ctx.fillText(`Required Orientation: ${POSITIONS[currentPosition]}`, 20, 90);

          // Display orientation scores if available
          if (Object.keys(orientationScores).length > 0) {
            ctx.fillText(`Orientation Scores: ${JSON.stringify(orientationScores).slice(0, 35)}...`, 20, 110);
          }

          // Draw countdown timer if active
          if (captureCountdown) {
            ctx.fillStyle = "rgba(74, 86, 226, 0.7)";
            ctx.font = "48px Arial";
            ctx.fillText(`${captureCountdown}`, canvas.width / 2 - 15, canvas.height / 2 + 15);
          }
        }
      }
    }
  }, [boundingBox, boundingBoxColor, cameraActive, detectionConfidence, lightingLevel, debugMode, currentOrientation, orientationScores, captureCountdown, currentPosition]);

  // Face detection and position verification
  useEffect(() => {
    if (!cameraActive || !webcamRef.current || !modelsLoaded) return;
    let detectionInterval;

    const detectFaces = async () => {
      if (!webcamRef.current?.video || webcamRef.current.video.readyState !== 4) return;

      try {
        const video = webcamRef.current.video;
        const detections = await faceapi
          .detectAllFaces(video, new faceapi.TinyFaceDetectorOptions({ scoreThreshold: 0.2 }))
          .withFaceLandmarks();

        const brightness = calculateBrightness(video);
        setLightingLevel(brightness);

        if (detections.length > 1) {
          setInstructions("Multiple faces detected. Only one face should be visible.");
          setBoundingBoxColor("#ff0000");
          setFaceDetected(false);
          const detection = detections[0];
          setBoundingBox(detection.detection.box);
          setDetectionConfidence(detection.detection.score);
          setCurrentOrientation("multiple");
          setConsecutiveCorrectOrientations(0);
        } else if (detections.length === 0) {
          setInstructions("No face detected. Position your face in the center.");
          setBoundingBoxColor("#ff0000");
          setFaceDetected(false);
          setBoundingBox(null);
          setDetectionConfidence(0);
          setCurrentOrientation("none");
          setConsecutiveCorrectOrientations(0);
        } else {
          setFaceDetected(true);
          const detection = detections[0];
          const box = detection.detection.box;
          setBoundingBox(box);
          setDetectionConfidence(detection.detection.score);

          // Get detailed face orientation with score values
          const { orientation, scores } = determineFaceOrientationWithScores(detection);
          setCurrentOrientation(orientation);
          setOrientationScores(scores);

          if (brightness < LIGHTING_THRESHOLD_POOR) {
            setInstructions("Low lighting detected. Try to improve lighting if possible.");
            setBoundingBoxColor("#ff9800");
          } else if (detection.detection.score < DETECTION_THRESHOLD_POOR) {
            setInstructions("Low detection quality. Try adjusting your position.");
            setBoundingBoxColor("#ff9800");
          } else {
            setBoundingBoxColor("#4caf50");
            const requiredPosition = POSITIONS[currentPosition];

            if (orientation === requiredPosition) {
              // Increment consecutive correct orientations counter
              setConsecutiveCorrectOrientations(prev => prev + 1);

              // Start countdown when we have enough consecutive correct orientations
              if (consecutiveCorrectOrientations >= 5 && !captureCountdown && !isProcessing) {
                setInstructions(`Great! Hold your ${requiredPosition} position...`);

                // Start countdown from 3
                setCaptureCountdown(3);
                const countdownInterval = setInterval(() => {
                  setCaptureCountdown(prev => {
                    if (prev <= 1) {
                      clearInterval(countdownInterval);
                      captureImage();
                      return null;
                    }
                    return prev - 1;
                  });
                }, 500);
              } else if (consecutiveCorrectOrientations < 5) {
                setInstructions(`Good! Keep your face ${requiredPosition}! (${consecutiveCorrectOrientations}/5)`);
              }
            } else {
              // Reset consecutive counter when orientation is wrong
              setConsecutiveCorrectOrientations(0);
              setCaptureCountdown(null);
              setInstructions(`Please turn your face ${requiredPosition} slightly!`);
            }
          }
        }
      } catch (error) {
        console.error("Error in face detection:", error);
      }
    };

    detectionInterval = setInterval(detectFaces, 200);
    return () => {
      clearInterval(detectionInterval);
      if (captureTimer) {
        clearTimeout(captureTimer);
        setCaptureTimer(null);
      }
    };
  }, [cameraActive, isProcessing, currentPosition, modelsLoaded, captureTimer, consecutiveCorrectOrientations, captureCountdown]);

  // Update current position when enough photos are taken for a position
  useEffect(() => {
    if (capturedImages.length > 0 && capturedImages.length % PHOTOS_PER_POSITION === 0) {
      const newPosition = Math.min(capturedImages.length / PHOTOS_PER_POSITION, POSITIONS.length - 1);
      if (newPosition !== currentPosition) {
        setCurrentPosition(newPosition);
        setInstructions(`Please turn your face ${POSITIONS[newPosition]}`);
        // Reset consecutive counter when moving to new position
        setConsecutiveCorrectOrientations(0);
      }

      if (capturedImages.length === TOTAL_PHOTOS_REQUIRED) {
        setInstructions("All photos captured successfully!");
        setCameraActive(false);
        setFaceRegistrationComplete(true);  // Set face registration to true
      }
    }
  }, [capturedImages.length, currentPosition]);

  // Enhanced face orientation detection with numerical scores
  const determineFaceOrientationWithScores = (detection) => {
    const landmarks = detection.landmarks;
    const jawOutline = landmarks.getJawOutline();
    const nose = landmarks.getNose();
    const leftEye = landmarks.getLeftEye();
    const rightEye = landmarks.getRightEye();

    const leftEyeCenter = getCenterPoint(leftEye);
    const rightEyeCenter = getCenterPoint(rightEye);
    const eyeLevel = (leftEyeCenter.y + rightEyeCenter.y) / 2;

    const noseCenter = getCenterPoint(nose);
    const jawCenter = getCenterPoint(jawOutline);

    // Improved width measurement using outer points of jaw
    const faceWidth = jawOutline[16].x - jawOutline[0].x;

    // Calculate normalized drift metrics
    const noseDrift = (noseCenter.x - jawCenter.x) / faceWidth;
    const eyeLevelRatio = eyeLevel / jawCenter.y;

    // Calculate eye distance ratio (helps with detecting rotation)
    const eyeDistance = Math.abs(rightEyeCenter.x - leftEyeCenter.x);
    const eyeDistanceRatio = eyeDistance / faceWidth;

    // Calculate scores for each orientation
    const scores = {
      front: 1 - (Math.abs(noseDrift) * 5) - (Math.abs(eyeLevelRatio - 1) * 5),
      right: noseDrift > 0 ? noseDrift * 5 : 0,
      left: noseDrift < 0 ? Math.abs(noseDrift) * 5 : 0
    };

    // Determine the most likely orientation using thresholds and scores
    let orientation;
    if (noseDrift < -ORIENTATION_THRESHOLD.NOSE_DRIFT_SIDE) {
      orientation = "left";
    } else if (noseDrift > ORIENTATION_THRESHOLD.NOSE_DRIFT_SIDE) {
      orientation = "right";
    } else {
      orientation = "front";
    }

    return { orientation, scores };
  };

  const getCenterPoint = (points) => {
    const sumX = points.reduce((sum, point) => sum + point.x, 0);
    const sumY = points.reduce((sum, point) => sum + point.y, 0);
    return {
      x: sumX / points.length,
      y: sumY / points.length
    };
  };

  const calculateBrightness = (video) => {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const data = imageData.data;

    let brightness = 0;
    let pixelCount = 0;

    // Sample fewer pixels for performance (every 12th pixel)
    for (let i = 0; i < data.length; i += 12) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      brightness += (r + g + b) / 3;
      pixelCount++;
    }

    return brightness / pixelCount;
  };

  const captureImage = () => {
    if (isProcessing || !webcamRef.current) return;

    setIsProcessing(true);
    const imageSrc = webcamRef.current.getScreenshot();

    setCapturedImages((prev) => [...prev, imageSrc]);
    setConsecutiveCorrectOrientations(0);

    setTimeout(() => {
      setIsProcessing(false);
    }, 500);
  };

  const toggleSection = (section, groupSections) => {
    setSelectedSections(prev => {
      const filtered = prev.filter(s => !groupSections.includes(s));
      return filtered.includes(section) ? filtered : [...filtered, section];
    });
  };

  const validateData = () => {
    if (!fullName || !username || !branch || !semester || !batch || selectedSections.length === 0) {
      alert("Please fill in all fields");
      return false;
    }
    if (password.length < 8 || password !== confirmPassword) {
      alert("Password must be at least 8 characters and match the confirmation");
      return false;
    }
    if (capturedImages.length < TOTAL_PHOTOS_REQUIRED) {
      alert(`Please capture all ${TOTAL_PHOTOS_REQUIRED} images`);
      return false;
    }
    return true;
  };

  const startFaceRegistration = () => {
    setCameraActive(true);
    setCapturedImages([]);
    setCurrentPosition(0);
    setInstructions("Please position your face front and center");
    setConsecutiveCorrectOrientations(0);
    setCaptureCountdown(null);
  };

  const handleRegister = async () => {
    // Validate data before proceeding
    if (!validateData()) {
      alert("Please fill in all required fields correctly.");
      return;
    }

    // Additional validation for passwords and photos
    if (password !== confirmPassword) {
      alert("Passwords do not match.");
      return;
    }
    if (capturedImages.length < TOTAL_PHOTOS_REQUIRED) {
      alert(`Please capture at least ${TOTAL_PHOTOS_REQUIRED} photos.`);
      return;
    }

    setLoading(true);

    try {
      const formData = new FormData();
      // Append form fields
      formData.append("roll_number", rollNumber);
      formData.append("full_name", fullName);
      formData.append("email", email);
      formData.append("branch", branch);
      formData.append("semester", parseInt(semester)); // Ensure integer
      formData.append("batch", parseInt(batch));       // Ensure integer
      formData.append("username", username);
      formData.append("password", password);
      formData.append("sections_selected", JSON.stringify(selectedSections));

      // Convert captured image URLs to blobs and append to FormData
      const imagePromises = capturedImages.map(async (img, index) => {
        const response = await fetch(img);
        if (!response.ok) throw new Error(`Failed to fetch image ${index + 1}`);
        const blob = await response.blob();
        formData.append("files", blob, `${rollNumber}_${index + 1}.jpg`);
        return true;
      });

      // Wait for all images to be processed
      await Promise.all(imagePromises);

      // Send the request to the backend
      const response = await axios.post(`${BASE_URL}/students/register`, formData, {
        headers: { "Content-Type": "multipart/form-data" }, // Note: axios sets this automatically for FormData
      });

      // Check for successful registration
      if (response.data.message) {
        alert("Registration completed successfully!");
        onSubmit(); // Call the parent callback to handle post-registration logic
      } else {
        throw new Error("Unexpected response from server");
      }
    } catch (error) {
      // Detailed error handling
      const errorMessage =
        error.response?.data?.detail || error.message || "Unknown error occurred";
      console.error("Registration failed:", error);
      alert(`Registration failed: ${errorMessage}`);
    } finally {
      setLoading(false); // Ensure loading state is reset
    }
  };

  // Helper function to get a color based on captured image progress
  const getProgressColor = () => {
    const progress = capturedImages.length / TOTAL_PHOTOS_REQUIRED;
    if (progress < 0.3) return "#ff9800";
    if (progress < 0.7) return "#2196f3";
    return "#4caf50";
  };

  return (
    <div style={styles.container}>
      <div style={styles.formContainer}>
        <h2 style={styles.title}>Complete Registration</h2>
        <input
          style={styles.input}
          placeholder="Full Name"
          value={fullName}
          onChange={(e) => setFullName(e.target.value)}
        />
        <input
          style={styles.input}
          placeholder="Username"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
        />
        <select
          style={styles.input}
          value={branch}
          onChange={(e) => setBranch(e.target.value)}
        >
          <option value="">Select Branch</option>
          {branchOptions.map((opt) => (
            <option key={opt} value={opt}>{opt}</option>
          ))}
        </select>
        <select
          style={styles.input}
          value={semester}
          onChange={(e) => setSemester(e.target.value)}
        >
          <option value="">Select Semester</option>
          {semesterOptions.map((opt) => (
            <option key={opt} value={opt}>{opt}</option>
          ))}
        </select>
        <select
          style={styles.input}
          value={batch}
          onChange={(e) => setBatch(e.target.value)}
        >
          <option value="">Select Batch</option>
          {batchOptions.map((opt) => (
            <option key={opt} value={opt}>{opt}</option>
          ))}
        </select>
        {availableSections.length > 0 && (
          <div style={styles.sectionsContainer}>
            <h3>Select Sections</h3>
            {availableSections.map((group) => (
              <div key={group.category} style={styles.sectionGroup}>
                <h4 style={styles.groupTitle}>{group.category}</h4>
                <div style={styles.sectionButtonsContainer}>
                  {group.sections.map((section) => (
                    <button
                      key={section}
                      style={{
                        ...styles.sectionButton,
                        backgroundColor: selectedSections.includes(section) ? "#4a56e2" : "#f0f2ff",
                        color: selectedSections.includes(section) ? "#fff" : "#4a56e2",
                      }}
                      onClick={() => toggleSection(section, group.sections)}
                    >
                      {section}
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        )}
        <input
          style={styles.input}
          type="password"
          placeholder="Password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
        />
        <input
          style={styles.input}
          type="password"
          placeholder="Confirm Password"
          value={confirmPassword}
          onChange={(e) => setConfirmPassword(e.target.value)}
        />
      </div>

      <div style={styles.cameraSection}>
        {!modelsLoaded && !modelLoadingError && (
          <div style={styles.loadingMessage}>
            <p>Loading face detection models...</p>
            <p>This may take a few seconds</p>
          </div>
        )}

        {modelLoadingError && (
          <div style={styles.errorMessage}>
            Failed to load face detection models. Please refresh the page.
          </div>
        )}

        <div style={styles.cameraButtons}>
          <button
            onClick={startFaceRegistration}
            disabled={!modelsLoaded || cameraActive || faceRegistrationComplete} // Modified Disabled Logic
            style={{
              ...styles.button,
              backgroundColor: (!modelsLoaded || cameraActive || faceRegistrationComplete) ? "#cccccc" : "#4a56e2", // Modified Styling
              width: "auto",
              flex: 2,
            }}
          >
            {faceRegistrationComplete ? "Face Registration Complete" : (modelsLoaded ? "Start Face Registration" : "Loading Models...")} {/* Modified Text */}
          </button>

          <button
            onClick={() => setDebugMode(!debugMode)}
            style={{
              ...styles.button,
              backgroundColor: "#5d6497",
              width: "auto",
              flex: 1,
              marginLeft: "10px",
            }}
          >
            {debugMode ? "Hide Debug Info" : "Show Debug Info"}
          </button>
        </div>

        {cameraActive && modelsLoaded && (
          <div style={styles.cameraContainer}>
            <div style={styles.webcamContainer}>
              <Webcam
                ref={webcamRef}
                style={styles.webcam}
                screenshotFormat="image/jpeg"
                audio={false}
                forceScreenshotSourceSize
                videoConstraints={{
                  width: 640,
                  height: 480,
                  facingMode: "user"
                }}
              />
              <canvas
                ref={canvasRef}
                style={styles.canvas}
              />
            </div>
          <p style={styles.cameraNote}>Your camera might be inverted, please take care of the orientation.<br/>
          </p>
            <div style={styles.cameraInfo}>

              <p style={{
                ...styles.instructions,
                color: boundingBoxColor === "#ff0000" ? "#ff0000" :
                  boundingBoxColor === "#ff9800" ? "#ff9800" : "#4a56e2"
              }}>
                {instructions}
              </p>

              <div style={styles.progressContainer}>
                <div style={styles.progressBar}>
                  <div
                    style={{
                      ...styles.progressFill,
                      width: `${(capturedImages.length / TOTAL_PHOTOS_REQUIRED) * 100}%`,
                      backgroundColor: getProgressColor()
                    }}
                  />
                </div>
                <p style={styles.progressText}>
                  {capturedImages.length}/{TOTAL_PHOTOS_REQUIRED} photos captured
                </p>
              </div>

              <div style={styles.metricsContainer}>
                <div style={styles.metric}>
                  <span>Face Position:</span>
                  <span>{POSITIONS[currentPosition]}</span>
                </div>
                <div style={styles.metric}>
                  <span>Current Orientation:</span>
                  <span style={{
                    color: currentOrientation === POSITIONS[currentPosition] ? "#4caf50" : "#ff9800"
                  }}>
                    {currentOrientation}
                  </span>
                </div>
                <div style={styles.metric}>
                  <span>Lighting:</span>
                  <span style={{
                    color: lightingLevel < LIGHTING_THRESHOLD_POOR ? "#ff0000" :
                      lightingLevel < LIGHTING_THRESHOLD_MODERATE ? "#ff9800" : "#4caf50"
                  }}>
                    {lightingLevel < LIGHTING_THRESHOLD_POOR ? "Poor" :
                      lightingLevel < LIGHTING_THRESHOLD_MODERATE ? "Moderate" : "Good"} ({lightingLevel.toFixed(1)})
                  </span>
                </div>
                <div style={styles.metric}>
                  <span>Detection Quality:</span>
                  <span style={{
                    color: detectionConfidence < DETECTION_THRESHOLD_POOR ? "#ff0000" :
                      detectionConfidence < DETECTION_THRESHOLD_MODERATE ? "#ff9800" : "#4caf50"
                  }}>
                    {detectionConfidence < DETECTION_THRESHOLD_POOR ? "Poor" :
                      detectionConfidence < DETECTION_THRESHOLD_MODERATE ? "Moderate" : "Good"} ({detectionConfidence.toFixed(2)})
                  </span>
                </div>
              </div>

            </div>
          </div>
        )}
      </div>

      <button
        onClick={handleRegister}
        disabled={loading || !faceRegistrationComplete} // Disable registration button if face registration not complete
        style={{
          ...styles.button,
          backgroundColor: (loading || !faceRegistrationComplete) ? "#cccccc" : "#4a56e2",
          marginTop: "20px"
        }}
      >
        {loading ? "Registering..." : "Complete Registration"}
      </button>
    </div>
  );
};

const styles = {
  container: {
    backgroundColor: "#ffffff",
    borderRadius: "15px",
    padding: "20px",
    width: "100%",
    maxWidth: "800px",
    boxShadow: "0 2px 10px rgba(0,0,0,0.1)",
  },
  formContainer: {
    marginBottom: "30px",
  },
  title: {
    fontSize: "24px",
    fontWeight: "bold",
    color: "#333",
    textAlign: "center",
    marginBottom: "30px",
  },
  input: {
    width: "100%",
    padding: "12px",
    marginBottom: "15px",
    border: "1px solid #e0e0e0",
    borderRadius: "10px",
    fontSize: "16px",
    boxSizing: "border-box",
  },
  sectionsContainer: {
    marginBottom: "20px",
  },
  sectionButtonsContainer: {
    display: "flex",
    flexWrap: "wrap",
  },
  sectionButton: {
    padding: "10px 15px",
    margin: "5px",
    borderRadius: "10px",
    border: "none",
    cursor: "pointer",
    fontSize: "14px",
    fontWeight: "500",
    boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
    transition: "all 0.2s ease",
  },
  sectionGroup: {
    marginBottom: "20px",
  },
  groupTitle: {
    fontSize: "14px",
    color: "#666",
    marginBottom: "10px",
  },
  cameraSection: {
    marginBottom: "20px",
  },
  cameraButtons: {
    display: "flex",
    marginBottom: "15px",
  },
  cameraContainer: {
    backgroundColor: "#f9f9f9",
    borderRadius: "10px",
    padding: "15px",
    boxShadow: "0 2px 6px rgba(0,0,0,0.05)",
  },
  webcamContainer: {
    position: "relative",
    width: "100%",
    borderRadius: "10px",
    overflow: "hidden",
    boxShadow: "0 2px 8px rgba(0,0,0,0.1)",
  },
  webcam: {
    width: "100%",
    height: "auto",
    borderRadius: "10px",
  },
  canvas: {
    position: "absolute",
    top: 0,
    left: 0,
    width: "100%",
    height: "100%",
  },
  cameraInfo: {
    marginTop: "15px",
  },
  instructions: {
    textAlign: "center",
    fontSize: "16px",
    fontWeight: "500",
    marginBottom: "15px",
  },
  progressContainer: {
    marginBottom: "15px",
  },
  progressBar: {
    width: "100%",
    height: "10px",
    backgroundColor: "#e0e0e0",
    borderRadius: "5px",
    overflow: "hidden",
  },
  progressFill: {
    height: "100%",
    backgroundColor: "#4a56e2",
    transition: "width 0.3s ease",
  },
  progressText: {
    textAlign: "center",
    marginTop: "10px",
    fontWeight: "bold",
    color: "#4a56e2",
  },
  metricsContainer: {
    display: "flex",
    justifyContent: "space-around",
    flexWrap: "wrap",
    gap: "10px",
    marginTop: "15px",
  },
  metric: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    backgroundColor: "#f0f2ff",
    padding: "8px 15px",
    borderRadius: "8px",
    minWidth: "100px",
  },
  button: {
    padding: "15px",
    borderRadius: "10px",
    textAlign: "center",
    fontSize: "16px",
    fontWeight: "bold",
    border: "none",
    cursor: "pointer",
    width: "100%",
    color: "#fff",
    transition: "background-color 0.3s ease",
  },
  loadingMessage: {
    padding: '20px',
    textAlign: 'center',
    color: '#666',
    backgroundColor: '#f5f5f5',
    borderRadius: '10px',
    marginBottom: '20px',
  },
  errorMessage: {
    color: '#ff0000',
    textAlign: 'center',
    padding: '10px',
    backgroundColor: '#ffeeee',
    borderRadius: '10px',
    marginBottom: '20px',
    fontWeight: 'bold',
  },
};

export default CompleteRegistration;