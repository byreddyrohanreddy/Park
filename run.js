/**
 * VoiceCare AI — Unified 1-Command System Launcher
 * Automatically coordinates:
 *   1. Flask ML Inference Engine (:5001) using the 'ml' conda environment
 *   2. Node.js & Express API Gateway (:5000)
 *   3. React & Vite Web Client (:5173)
 */

const { spawn, execSync } = require('child_process');
const path = require('path');
const os = require('os');
const fs = require('fs');

const ROOT_DIR = __dirname;
const ML_DIR = path.join(ROOT_DIR, 'backend', 'ml');
const FRONTEND_DIR = path.join(ROOT_DIR, 'frontend');

// Terminal colors
const RESET = '\x1b[0m';
const BOLD = '\x1b[1m';
const GREEN = '\x1b[32m';
const BLUE = '\x1b[34m';
const MAGENTA = '\x1b[35m';
const YELLOW = '\x1b[33m';
const RED = '\x1b[31m';
const CYAN = '\x1b[36m';

console.log(`${BOLD}${CYAN}==============================================================${RESET}`);
console.log(`${BOLD}${CYAN}    🧠 VoiceCare AI — PD-VoiceNet Unified Launcher           ${RESET}`);
console.log(`${BOLD}${CYAN}==============================================================${RESET}\n`);

// 1. Locate Python executable in the 'ml' conda environment
function findPython() {
  if (process.env.PYTHON && fs.existsSync(process.env.PYTHON)) {
    console.log(`${GREEN}✔ Using explicit PYTHON:${RESET} ${process.env.PYTHON}`);
    return { cmd: process.env.PYTHON, args: [] };
  }

  const userHome = os.homedir();
  const candidateMlPaths = [
    path.join(userHome, 'miniconda3', 'envs', 'ml', 'python.exe'),
    path.join(userHome, 'anaconda3', 'envs', 'ml', 'python.exe'),
    path.join(userHome, 'AppData', 'Local', 'miniconda3', 'envs', 'ml', 'python.exe'),
    path.join(userHome, 'AppData', 'Local', 'anaconda3', 'envs', 'ml', 'python.exe'),
    'C:\\miniconda3\\envs\\ml\\python.exe',
    'C:\\anaconda3\\envs\\ml\\python.exe',
    path.join(userHome, 'miniconda3', 'envs', 'voicecare', 'python.exe'),
  ];

  for (const p of candidateMlPaths) {
    if (fs.existsSync(p)) {
      console.log(`${GREEN}✔ Found 'ml' conda environment:${RESET} ${p}`);
      return { cmd: p, args: [] };
    }
  }

  // Check if conda is available in PATH
  try {
    execSync('conda --version', { stdio: 'ignore' });
    console.log(`${GREEN}✔ Using 'conda run -n ml python'${RESET}`);
    return { cmd: 'conda', args: ['run', '-n', 'ml', '--no-capture-output', 'python'] };
  } catch (e) {}

  // Fallback to system python
  console.log(`${YELLOW}⚠ Conda 'ml' path not directly found. Falling back to system 'python'...${RESET}`);
  return { cmd: 'python', args: [] };
}

// 2. Kill old processes on ports 5000 and 5001 if left hanging from past runs (Windows)
function clearPorts(ports) {
  if (os.platform() !== 'win32') return;
  for (const port of ports) {
    try {
      const stdout = execSync(`netstat -ano | findstr :${port}`, { encoding: 'utf8', stdio: ['ignore', 'pipe', 'ignore'] });
      const lines = stdout.trim().split('\n');
      for (const line of lines) {
        if (line.includes('LISTENING')) {
          const parts = line.trim().split(/\s+/);
          const pid = parts[parts.length - 1];
          if (pid && pid !== '0' && pid !== String(process.pid)) {
            console.log(`${YELLOW}Freeing port ${port} (killing leftover PID ${pid})...${RESET}`);
            execSync(`taskkill /F /PID ${pid}`, { stdio: 'ignore' });
          }
        }
      }
    } catch (e) {}
  }
}

clearPorts([5000, 5001]);

const pythonInfo = findPython();
const children = [];

function prefixPipe(stream, prefix, color) {
  if (!stream) return;
  let buffer = '';
  stream.on('data', (data) => {
    buffer += data.toString();
    const lines = buffer.split('\n');
    buffer = lines.pop(); // keep partial line in buffer
    for (const line of lines) {
      if (line.trim().length > 0) {
        console.log(`${color}${BOLD}[${prefix}]${RESET} ${line}`);
      }
    }
  });
}

function startProcess(name, cmd, args, cwd, color) {
  const isWindows = os.platform() === 'win32';
  const child = spawn(cmd, args, {
    cwd,
    shell: isWindows,
    env: { ...process.env, PYTHONUNBUFFERED: '1', FORCE_COLOR: '1' }
  });

  prefixPipe(child.stdout, name, color);
  prefixPipe(child.stderr, name, RED);

  child.on('error', (err) => {
    console.error(`${RED}${BOLD}[${name} ERROR]${RESET} ${err.message}`);
  });

  child.on('exit', (code, signal) => {
    if (code !== 0 && code !== null) {
      console.log(`${YELLOW}[${name}] exited with code ${code}${RESET}`);
    }
  });

  children.push({ name, process: child });
  return child;
}

// 3. Start Python Flask ML Server (port 5001)
const mlArgs = [...pythonInfo.args, 'app.py'];
startProcess('ML-API:5001', pythonInfo.cmd, mlArgs, ML_DIR, GREEN);

// 4. Start Node.js Express Server (port 5000)
startProcess('NODE:5000', 'node', ['backend/server.js'], ROOT_DIR, BLUE);

// 5. Start Vite Frontend (port 5173)
const npmCmd = os.platform() === 'win32' ? 'npm.cmd' : 'npm';
startProcess('VITE:5173', npmCmd, ['run', 'dev'], FRONTEND_DIR, MAGENTA);

console.log(`\n${BOLD}${GREEN}🚀 All 3 services are launching!${RESET}`);
console.log(`   ${CYAN}• Frontend UI:${RESET}       ${BOLD}http://localhost:5173${RESET}`);
console.log(`   ${CYAN}• Node Express API:${RESET}  ${BOLD}http://localhost:5000${RESET}`);
console.log(`   ${CYAN}• Python ML Engine:${RESET}  ${BOLD}http://127.0.0.1:5001/health${RESET}`);
console.log(`\n${YELLOW}Press ${BOLD}Ctrl + C${RESET}${YELLOW} to stop all services simultaneously.\n${RESET}`);

// Clean shutdown handler
let isShuttingDown = false;
function shutdown() {
  if (isShuttingDown) return;
  isShuttingDown = true;
  console.log(`\n${YELLOW}Shutting down all VoiceCare AI services...${RESET}`);

  for (const { name, process: child } of children) {
    try {
      if (os.platform() === 'win32' && child.pid) {
        execSync(`taskkill /F /T /PID ${child.pid}`, { stdio: 'ignore' });
      } else {
        child.kill('SIGTERM');
      }
    } catch (e) {}
  }
  setTimeout(() => process.exit(0), 500);
}

process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
process.on('exit', shutdown);
