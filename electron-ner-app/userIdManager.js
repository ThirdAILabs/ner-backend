import { app } from 'electron';
import path from 'node:path';
import fs from 'node:fs';
import crypto from 'node:crypto';

const USER_ID_FILE = 'user-id.json';

// Get the path where we'll store the user ID
function getUserIdFilePath() {
  const userDataPath = app.getPath('userData');
  return path.join(userDataPath, USER_ID_FILE);
}

// Generate a pseudonymous user ID (UUID-like but shorter)
function generateUserId() {
  const randomBytes = crypto.randomBytes(8);
  const timestamp = Date.now().toString(36);
  const random = randomBytes.toString('hex');
  return `user_${timestamp}_${random}`;
}

// Load existing user ID or generate a new one
export async function getOrCreateUserId() {
  const filePath = getUserIdFilePath();
  
  try {
    // Try to read existing user ID
    if (fs.existsSync(filePath)) {
      const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
      if (data.userId && typeof data.userId === 'string') {
        console.log('Loaded existing user ID:', data.userId);
        return data.userId;
      }
    }
  } catch (error) {
    console.warn('Error reading user ID file:', error);
  }

  // Generate new user ID
  const newUserId = generateUserId();
  
  try {
    // Ensure the user data directory exists
    const userDataPath = app.getPath('userData');
    if (!fs.existsSync(userDataPath)) {
      fs.mkdirSync(userDataPath, { recursive: true });
    }
    
    // Save the new user ID
    const data = {
      userId: newUserId,
      createdAt: new Date().toISOString(),
      version: '1.0.0'
    };
    
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2), 'utf8');
    console.log('Generated and saved new user ID:', newUserId);
    return newUserId;
  } catch (error) {
    console.error('Error saving user ID:', error);
    // Fall back to session-only ID if we can't save
    console.warn('Using session-only user ID:', newUserId);
    return newUserId;
  }
}

// Get the current user ID (assumes it's already been loaded)
let currentUserId = null;

export async function initializeUserId() {
  currentUserId = await getOrCreateUserId();
  return currentUserId;
}

export function getCurrentUserId() {
  return currentUserId || 'anonymous';
} 