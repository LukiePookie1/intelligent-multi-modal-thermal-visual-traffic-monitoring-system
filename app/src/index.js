// main.js
const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

if (require('electron-squirrel-startup')) {
  app.quit();
}

let mainWindow;

const createWindow = () => {
  mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true,
    },
  });
  mainWindow.loadFile(path.join(__dirname, 'index.html'));
  mainWindow.webContents.openDevTools();
};

app.on('ready', createWindow);

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});

ipcMain.on('process-video', (event, inputVideoPath, outputVideoPath) => {
  console.log('Received process-video event with paths:', inputVideoPath, outputVideoPath);
  const pythonProcess = spawn('python', ['object-detect-script.py', inputVideoPath, outputVideoPath]);

  pythonProcess.stdout.on('data', (data) => {
    console.log(`Python script output: ${data}`);
    mainWindow.webContents.send('processing-update', data.toString());
  });

  pythonProcess.stderr.on('data', (data) => {
    console.error(`Python script error: ${data}`);
  });

  pythonProcess.on('close', (code) => {
    console.log(`Python script exited with code ${code}`);
    mainWindow.webContents.send('processing-complete', outputVideoPath);
  });
});