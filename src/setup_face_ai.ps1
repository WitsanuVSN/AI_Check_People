# Run from the ai_check_people project folder.
$ErrorActionPreference = "Stop"

Write-Host "1) Installing MediaPipe..." -ForegroundColor Cyan
npm install @mediapipe/tasks-vision

Write-Host "2) Creating model and wasm folders..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force .\public\models | Out-Null
New-Item -ItemType Directory -Force .\public\wasm | Out-Null

Write-Host "3) Copying WASM files from node_modules..." -ForegroundColor Cyan
Copy-Item .\node_modules\@mediapipe\tasks-vision\wasm\* .\public\wasm\ -Recurse -Force

Write-Host "4) Downloading full-range face detection model..." -ForegroundColor Cyan
Invoke-WebRequest `
  -Uri "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_full_range/float16/1/blaze_face_full_range.tflite" `
  -OutFile ".\public\models\face_detector_full_range.tflite"

Write-Host ""
Write-Host "Checking files:" -ForegroundColor Green
Get-Item .\public\models\face_detector_full_range.tflite | Select-Object Name,Length
Get-ChildItem .\public\wasm | Select-Object Name,Length

Write-Host ""
Write-Host "Done. Run: npm run dev" -ForegroundColor Green
