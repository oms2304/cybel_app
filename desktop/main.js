const { app, BrowserWindow } = require("electron");
const path = require("path");
const { spawn } = require("child_process");
const waitOn = require("wait-on");

let backendProcess = null;

function startBackend() {
  const backendDir = path.join(__dirname, "..", "backend");

  backendProcess = spawn(
    "python",
    ["-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000"],
    {
      cwd: backendDir,
      stdio: "inherit",
      windowsHide: false,
    }
  );

  backendProcess.on("close", (code) => {
    console.log("Backend exited with code:", code);
  });
}

async function createWindow() {
    await waitOn({
    resources: ["http-get://127.0.0.1:8000/health"],
    timeout: 60000,
    interval: 250,
    });

  const win = new BrowserWindow({
    width: 1080,
    height: 780,
  });

  const htmlPath = path.join(
    __dirname,
    "..",
    "frontend",
    "Cybel Dashboard.html"
  );

  await win.loadFile(htmlPath);
}

app.whenReady().then(async () => {
  startBackend();
  await createWindow();
});

app.on("window-all-closed", () => {
  if (backendProcess) backendProcess.kill();
  if (process.platform !== "darwin") app.quit();
});
