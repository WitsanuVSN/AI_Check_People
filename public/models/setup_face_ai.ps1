$ErrorActionPreference = "Stop"

Write-Host "1) Creating models folder..." -ForegroundColor Cyan
New-Item -ItemType Directory -Force .\public\models | Out-Null

Write-Host "2) Copying face-api models from node_modules..." -ForegroundColor Cyan
Copy-Item .\node_modules\@vladmandic\face-api\model\* .\public\models\ -Recurse -Force

Write-Host ""
Write-Host "Done. Run: npm run dev" -ForegroundColor Green