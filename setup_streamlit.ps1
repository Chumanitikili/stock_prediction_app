# Define variables
$repoName = "stock_prediction_app"
$username = "Chumanitikili"
$mainFileName = "streamlit_app.py"
$mainFilePath = "C:\Users\Chumani\stock_prediction_app\app.py"
$branchName = "main"

# Create and navigate to temporary directory
$tempDir = New-Item -ItemType Directory -Path "$env:TEMP\$repoName" -Force
Set-Location $tempDir

Write-Host "Initializing Git repository..." -ForegroundColor Green
git init

# Check if main file exists, if not create a sample one
if (Test-Path $mainFilePath) {
    Write-Host "Copying your existing app file to $mainFileName..." -ForegroundColor Green
    Copy-Item $mainFilePath -Destination "$tempDir\$mainFileName"
} else {
    Write-Host "Creating a sample Streamlit app file..." -ForegroundColor Green
    @"
import streamlit as st
st.title('Stock Prediction App')
st.write('Welcome to the best Stock Prediction Application!')
"@ | Out-File -FilePath "$mainFileName" -Encoding utf8
}

# Create requirements.txt
Write-Host "Creating requirements.txt file..." -ForegroundColor Green
@"
streamlit==1.24.0
pandas==1.5.3
numpy==1.24.2
scikit-learn==1.2.2
matplotlib==3.7.1
"@ | Out-File -FilePath "requirements.txt" -Encoding utf8

# Git operations
Write-Host "Adding files to Git..." -ForegroundColor Green
git add .

Write-Host "Committing changes..." -ForegroundColor Green
git commit -m "Initial commit for Streamlit deployment"

# Explicitly set the main branch
Write-Host "Setting branch to $branchName..." -ForegroundColor Green
git branch -M "$branchName"

Write-Host "Adding remote repository..." -ForegroundColor Green
git remote add origin "https://github.com/$username/$repoName.git"

Write-Host "Pushing to GitHub..." -ForegroundColor Green
Write-Host "You will be prompted to enter your GitHub credentials..." -ForegroundColor Yellow
git push -u origin "$branchName"

# Display completion message
Write-Host "`nRepository setup complete!" -ForegroundColor Green
Write-Host "GitHub Repository: https://github.com/$username/$repoName.git" -ForegroundColor Cyan
Write-Host "Branch: $branchName" -ForegroundColor Cyan
Write-Host "Main file: $mainFileName" -ForegroundColor Cyan
Write-Host "`nNow go to Streamlit Cloud and deploy using these details." -ForegroundColor Magenta