const { exec } = require('child_process');
const net = require('net');

// Choose a rare 5-digit port for our application
const FIXED_PORT = 16549;

/**
 * Checks if the specified port is in use
 * @param {number} port Port to check
 * @returns {Promise<boolean>} True if port is in use, false otherwise
 */
function isPortInUse(port) {
  return new Promise((resolve) => {
    const server = net.createServer();
    
    server.once('error', (err) => {
      if (err.code === 'EADDRINUSE') {
        resolve(true); // Port is in use
      } else {
        resolve(false); // Some other error occurred
      }
    });
    
    server.once('listening', () => {
      // Close the server as we only wanted to check the port
      server.close();
      resolve(false); // Port is free
    });
    
    server.listen(port);
  });
}

/**
 * Finds and kills any process using the specified port
 * @param {number} port Port to free up
 * @returns {Promise<boolean>} True if successfully killed a process, false otherwise
 */
function killProcessOnPort(port) {
  return new Promise((resolve, reject) => {
    let command;
    if (process.platform === 'win32') {
      // Windows command
      command = `FOR /F "tokens=5" %P IN ('netstat -ano ^| findstr :${port} ^| findstr LISTENING') DO TaskKill /PID %P /F`;
    } else {
      // macOS/Linux command
      command = `lsof -i :${port} | grep LISTEN | awk '{print $2}' | xargs kill -9`;
    }

    exec(command, (error, stdout, stderr) => {
      if (error) {
        console.log(`No process found on port ${port} or could not kill: ${error.message}`);
        resolve(false);
      } else {
        console.log(`Successfully killed process on port ${port}`);
        resolve(true);
      }
    });
  });
}

/**
 * Ensures the designated port is free for use
 * @returns {Promise<boolean>} True if port is available (either was free or was successfully freed)
 */
async function ensurePortIsFree() {
  console.log(`Checking if port ${FIXED_PORT} is in use...`);
  
  const inUse = await isPortInUse(FIXED_PORT);
  
  if (inUse) {
    console.log(`Port ${FIXED_PORT} is in use, attempting to kill the process...`);
    const killed = await killProcessOnPort(FIXED_PORT);
    
    if (killed) {
      // Double check if port is now free
      const stillInUse = await isPortInUse(FIXED_PORT);
      if (stillInUse) {
        console.error(`Failed to free up port ${FIXED_PORT}!`);
        return false;
      }
      return true;
    } else {
      console.error(`Failed to kill process on port ${FIXED_PORT}!`);
      return false;
    }
  }
  
  console.log(`Port ${FIXED_PORT} is available.`);
  return true;
}

module.exports = {
  FIXED_PORT,
  isPortInUse,
  killProcessOnPort,
  ensurePortIsFree
};

// If this script is run directly
if (require.main === module) {
  ensurePortIsFree().then(success => {
    if (success) {
      console.log('Port is now available for use.');
    } else {
      console.error('Could not ensure port is available.');
      process.exit(1);
    }
  });
} 