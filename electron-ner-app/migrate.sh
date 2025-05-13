#!/bin/bash

# Create directories for components and other assets
mkdir -p src/components
mkdir -p src/hooks
mkdir -p src/lib
mkdir -p src/types
mkdir -p src/app

# Copy components
echo "Copying components..."
cp -R ../frontend/components/* src/components/

# Copy hooks if they exist
echo "Copying hooks..."
cp -R ../frontend/hooks/* src/hooks/

# Copy lib folder if it exists
echo "Copying lib..."
cp -R ../frontend/lib/* src/lib/

# Copy types
echo "Copying types..."
cp ../frontend/types.d.ts src/types/

# Copy app directory contents (excluding layout and page related files)
echo "Copying app contents..."
cp -R ../frontend/app/* src/app/

echo "Migration completed!" 