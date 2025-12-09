/**
 * Icon Generation Script for Futurnal Desktop Shell
 *
 * Generates all required icon sizes from the source logo.png
 * Required sizes: 32x32, 128x128, 128x128@2x (256x256), plus .icns and .ico
 */

import sharp from 'sharp';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const SOURCE = path.join(__dirname, '../../assets/logo.png');
const OUTPUT_DIR = path.join(__dirname, '../src-tauri/icons');

// PNG sizes for Tauri
const PNG_SIZES = [
  { name: '32x32.png', size: 32 },
  { name: '128x128.png', size: 128 },
  { name: '128x128@2x.png', size: 256 },
  { name: 'icon.png', size: 512 },
  // Windows Store logos
  { name: 'Square30x30Logo.png', size: 30 },
  { name: 'Square44x44Logo.png', size: 44 },
  { name: 'Square71x71Logo.png', size: 71 },
  { name: 'Square89x89Logo.png', size: 89 },
  { name: 'Square107x107Logo.png', size: 107 },
  { name: 'Square142x142Logo.png', size: 142 },
  { name: 'Square150x150Logo.png', size: 150 },
  { name: 'Square284x284Logo.png', size: 284 },
  { name: 'Square310x310Logo.png', size: 310 },
  { name: 'StoreLogo.png', size: 50 },
];

// ICNS sizes for macOS (in order of size)
const ICNS_SIZES = [16, 32, 64, 128, 256, 512, 1024];

// ICO sizes for Windows
const ICO_SIZES = [16, 24, 32, 48, 64, 128, 256];

async function generatePNGs() {
  console.log('Generating PNG icons...');

  for (const { name, size } of PNG_SIZES) {
    const outputPath = path.join(OUTPUT_DIR, name);
    await sharp(SOURCE)
      .resize(size, size, {
        fit: 'contain',
        background: { r: 0, g: 0, b: 0, alpha: 0 }
      })
      .png()
      .toFile(outputPath);
    console.log(`  Generated ${name} (${size}x${size})`);
  }
}

async function generateICNS() {
  console.log('Generating macOS .icns...');

  // Create iconset directory
  const iconsetDir = path.join(OUTPUT_DIR, 'icon.iconset');
  if (!fs.existsSync(iconsetDir)) {
    fs.mkdirSync(iconsetDir, { recursive: true });
  }

  // Generate all required sizes for iconset
  const iconsetSizes = [
    { name: 'icon_16x16.png', size: 16 },
    { name: 'icon_16x16@2x.png', size: 32 },
    { name: 'icon_32x32.png', size: 32 },
    { name: 'icon_32x32@2x.png', size: 64 },
    { name: 'icon_128x128.png', size: 128 },
    { name: 'icon_128x128@2x.png', size: 256 },
    { name: 'icon_256x256.png', size: 256 },
    { name: 'icon_256x256@2x.png', size: 512 },
    { name: 'icon_512x512.png', size: 512 },
    { name: 'icon_512x512@2x.png', size: 1024 },
  ];

  for (const { name, size } of iconsetSizes) {
    await sharp(SOURCE)
      .resize(size, size, {
        fit: 'contain',
        background: { r: 0, g: 0, b: 0, alpha: 0 }
      })
      .png()
      .toFile(path.join(iconsetDir, name));
  }

  // Try to use iconutil on macOS to create .icns
  try {
    const { execSync } = await import('child_process');
    execSync(`iconutil -c icns "${iconsetDir}" -o "${path.join(OUTPUT_DIR, 'icon.icns')}"`, {
      stdio: 'inherit'
    });
    console.log('  Generated icon.icns using iconutil');

    // Clean up iconset directory
    fs.rmSync(iconsetDir, { recursive: true });
  } catch (error) {
    console.log('  iconutil not available (not on macOS?). Using existing .icns or placeholder.');
    // Keep iconset for manual conversion
  }
}

async function generateICO() {
  console.log('Generating Windows .ico...');

  // Generate multi-size ICO using sharp
  // ICO format requires specific sizes
  const icoImages = [];

  for (const size of ICO_SIZES) {
    const buffer = await sharp(SOURCE)
      .resize(size, size, {
        fit: 'contain',
        background: { r: 0, g: 0, b: 0, alpha: 0 }
      })
      .png()
      .toBuffer();
    icoImages.push(buffer);
  }

  // Create ICO file manually
  // ICO header: 6 bytes
  // ICO directory entry: 16 bytes each
  // Image data follows

  const headerSize = 6;
  const directoryEntrySize = 16;
  const numImages = icoImages.length;

  let offset = headerSize + (directoryEntrySize * numImages);
  const entries = [];

  for (let i = 0; i < numImages; i++) {
    const size = ICO_SIZES[i];
    const imageData = icoImages[i];

    entries.push({
      width: size >= 256 ? 0 : size,  // 0 = 256
      height: size >= 256 ? 0 : size,
      colorCount: 0,
      reserved: 0,
      planes: 1,
      bitCount: 32,
      size: imageData.length,
      offset: offset
    });

    offset += imageData.length;
  }

  // Write ICO file
  const icoBuffer = Buffer.alloc(offset);
  let pos = 0;

  // Header
  icoBuffer.writeUInt16LE(0, pos); pos += 2;  // Reserved
  icoBuffer.writeUInt16LE(1, pos); pos += 2;  // Type (1 = ICO)
  icoBuffer.writeUInt16LE(numImages, pos); pos += 2;  // Number of images

  // Directory entries
  for (const entry of entries) {
    icoBuffer.writeUInt8(entry.width, pos); pos += 1;
    icoBuffer.writeUInt8(entry.height, pos); pos += 1;
    icoBuffer.writeUInt8(entry.colorCount, pos); pos += 1;
    icoBuffer.writeUInt8(entry.reserved, pos); pos += 1;
    icoBuffer.writeUInt16LE(entry.planes, pos); pos += 2;
    icoBuffer.writeUInt16LE(entry.bitCount, pos); pos += 2;
    icoBuffer.writeUInt32LE(entry.size, pos); pos += 4;
    icoBuffer.writeUInt32LE(entry.offset, pos); pos += 4;
  }

  // Image data
  for (const imageData of icoImages) {
    imageData.copy(icoBuffer, pos);
    pos += imageData.length;
  }

  fs.writeFileSync(path.join(OUTPUT_DIR, 'icon.ico'), icoBuffer);
  console.log('  Generated icon.ico');
}

async function main() {
  console.log('Futurnal Icon Generator');
  console.log('=======================');
  console.log(`Source: ${SOURCE}`);
  console.log(`Output: ${OUTPUT_DIR}`);
  console.log('');

  // Check source exists
  if (!fs.existsSync(SOURCE)) {
    console.error(`Error: Source file not found: ${SOURCE}`);
    process.exit(1);
  }

  // Ensure output directory exists
  if (!fs.existsSync(OUTPUT_DIR)) {
    fs.mkdirSync(OUTPUT_DIR, { recursive: true });
  }

  try {
    await generatePNGs();
    await generateICNS();
    await generateICO();

    console.log('');
    console.log('Icon generation complete!');
  } catch (error) {
    console.error('Error generating icons:', error);
    process.exit(1);
  }
}

main();
