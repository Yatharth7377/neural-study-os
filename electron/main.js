const { app, BrowserWindow, Menu } = require('electron');
const path = require('path');
const { spawn } = require('child_process');

let mainWindow;
let pythonProcess;

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 1000,
    minHeight: 600,
    backgroundColor: '#0a0a0f',
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true
    },
    titleBarStyle: 'hidden',
    title: 'Neural Study OS'
  });
  mainWindow.loadURL('http://localhost:5000');
  mainWindow.on('closed', () => { mainWindow = null; });
  const menu = Menu.buildFromTemplate([
    { label: 'Neural Study OS', submenu: [{ role: 'quit' }] },
    { label: 'Edit', submenu: [{ role: 'undo' }, { role: 'redo' }, { type: 'separator' }, { role: 'cut' }, { role: 'copy' }, { role: 'paste' }, { role: 'selectAll' }] },
    { label: 'View', submenu: [{ role: 'reload' }, { role: 'toggleDevTools' }, { type: 'separator' }, { role: 'zoomIn' }, { role: 'zoomOut' }, { role: 'resetZoom' }] }
  ]);
  Menu.setApplicationMenu(menu);
}

function startPython() {
  console.log('Starting Python backend...');
  const isWin = process.platform === 'win32';
  pythonProcess = spawn(isWin ? 'python' : 'python3', ['main.py']);
  pythonProcess.stdout.on('data', (data) => console.log('[Python]', data.toString()));
  pythonProcess.stderr.on('data', (data) => console.error('[Python Error]', data.toString()));
  pythonProcess.on('close', (code) => console.log('Python exited with code', code));
}

app.whenReady().then(() => {
  startPython();
  setTimeout(createWindow, 3000);
});

app.on('window-all-closed', () => {
  if (pythonProcess) pythonProcess.kill();
  if (process.platform !== 'darwin') app.quit();
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) createWindow();
});
