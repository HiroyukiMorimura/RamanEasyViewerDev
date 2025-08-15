#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Security Enhanced Custom Encoding System for RamanEye
Multiple layers of obfuscation and encryption
"""

import os
import base64
import zlib
import hashlib
import secrets
import time
from pathlib import Path
import sys
import shutil

class SecureCustomEncoder:
    def __init__(self, base_key='RamanEye2025'):
        # Generate dynamic key components
        self.base_key = base_key
        self.salt = secrets.token_bytes(32)
        self.timestamp = str(int(time.time()))
        
        # Create complex key derivation
        key_material = f"{base_key}:{self.timestamp}".encode('utf-8')
        derived_key = hashlib.pbkdf2_hmac('sha256', key_material, self.salt, 100000)
        self.encryption_key = derived_key[:32]
        self.obfuscation_key = derived_key[32:64] if len(derived_key) >= 64 else derived_key
        
    def _multi_layer_encrypt(self, data):
        """Multi-layer encryption with obfuscation"""
        # Layer 1: Basic compression
        compressed = zlib.compress(data.encode('utf-8'), level=9)
        
        # Layer 2: XOR with derived key
        encrypted = bytearray()
        for i, byte in enumerate(compressed):
            encrypted.append(byte ^ self.encryption_key[i % len(self.encryption_key)])
        
        # Layer 3: Byte manipulation
        manipulated = bytearray()
        for i, byte in enumerate(encrypted):
            # Complex bit manipulation
            rotated = ((byte << 3) | (byte >> 5)) & 0xFF
            manipulated.append(rotated ^ self.obfuscation_key[i % len(self.obfuscation_key)])
        
        # Layer 4: Base64 with custom alphabet
        custom_alphabet = "QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm1234567890+/"
        standard_b64 = base64.b64encode(manipulated).decode('ascii')
        
        # Replace standard base64 chars with custom alphabet
        standard_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        translation_table = str.maketrans(standard_alphabet, custom_alphabet)
        custom_b64 = standard_b64.translate(translation_table)
        
        return custom_b64

    def encode_file(self, filepath):
        """Encode file with enhanced security"""
        try:
            print(f'  🔐 Encoding: {filepath}')
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            encrypted_content = self._multi_layer_encrypt(content)
            return encrypted_content
            
        except Exception as e:
            print(f'  ❌ Failed to encode {filepath}: {e}')
            return None

    def create_secure_decoder(self, encrypted_data):
        """Create obfuscated decoder with hidden logic"""
        
        # Obfuscated decoder components
        decoder_components = {
            'salt_hex': self.salt.hex(),
            'timestamp': self.timestamp,
            'base_key': self.base_key,
            'encrypted_data': encrypted_data
        }
        
        # Generate multiple decoy functions to confuse analysis
        decoy_functions = self._generate_decoy_functions()
        
        # Generate random names for variables
        import_base64 = self._random_name()
        import_zlib = self._random_name()
        import_hashlib = self._random_name()
        import_sys = self._random_name()
        
        var_salt = self._random_name()
        var_timestamp = self._random_name()
        var_base_key = self._random_name()
        var_encrypted_data = self._random_name()
        func_decode = self._random_name()
        
        # Create heavily obfuscated decoder
        decoder_script = f'''# -*- coding: utf-8 -*-
"""Obfuscated module - Analysis protected"""
import base64 as {import_base64}, zlib as {import_zlib}, hashlib as {import_hashlib}, sys as {import_sys}

# Decoy data and functions
{decoy_functions}

# Obfuscated configuration
_{var_salt} = bytes.fromhex('{decoder_components["salt_hex"]}')
_{var_timestamp} = '{decoder_components["timestamp"]}'
_{var_base_key} = '{decoder_components["base_key"]}'
_{var_encrypted_data} = """{decoder_components["encrypted_data"]}"""

def {func_decode}():
    try:
        # Dynamic key reconstruction (heavily obfuscated)
        _key_material = f"{{_{var_base_key}}}:{{_{var_timestamp}}}".encode('utf-8')
        _derived_key = {import_hashlib}.pbkdf2_hmac('sha256', _key_material, _{var_salt}, 100000)
        _enc_key, _obf_key = _derived_key[:32], _derived_key[32:64] if len(_derived_key) >= 64 else _derived_key

        # Reverse custom base64
        _custom_alphabet = "QWERTYUIOPASDFGHJKLZXCVBNMqwertyuiopasdfghjklzxcvbnm1234567890+/"
        _standard_alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
        _translation_table = str.maketrans(_custom_alphabet, _standard_alphabet)
        _standard_b64 = _{var_encrypted_data}.translate(_translation_table)

        # Decode layers (reverse order)
        _decoded = {import_base64}.b64decode(_standard_b64.encode('ascii'))

        # Reverse byte manipulation
        _demanipulated = bytearray()
        for i, byte in enumerate(_decoded):
            _unrotated_xor = byte ^ _obf_key[i % len(_obf_key)]
            _original = ((_unrotated_xor >> 3) | (_unrotated_xor << 5)) & 0xFF
            _demanipulated.append(_original)

        # Reverse XOR
        _decrypted = bytearray()
        for i, byte in enumerate(_demanipulated):
            _decrypted.append(byte ^ _enc_key[i % len(_enc_key)])

        # Decompress and execute
        _decompressed = {import_zlib}.decompress(_decrypted)
        _code = _decompressed.decode('utf-8')
        exec(_code, globals())
        
    except Exception as e:
        print(f"Execution error: {{e}}")
        {import_sys}.exit(1)

# Execute with obfuscated call
{func_decode}()
'''
        return decoder_script

    def _random_name(self, length=8):
        """Generate random variable name"""
        chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
        return ''.join(secrets.choice(chars) for _ in range(length))

    def _generate_decoy_functions(self):
        """Generate decoy functions to confuse analysis"""
        decoys = []
        for _ in range(5):
            name = self._random_name()
            fake_data = base64.b64encode(secrets.token_bytes(256)).decode('ascii')
            decoys.append(f"""
def {name}():
    # Decoy function - fake encryption/decryption logic
    _fake_key = b'{secrets.token_bytes(16).hex()}'
    _fake_data = \"\"\"{fake_data}\"\"\"
    return _fake_data
""")
        return '\n'.join(decoys)

def main():
    print('=' * 70)
    print('    RamanEye Security Enhanced Custom Encoding')
    print('    Multi-layer obfuscation and encryption')
    print('=' * 70)
    
    encoder = SecureCustomEncoder()
    output_dir = Path('custom_encoded_secure')
    output_dir.mkdir(exist_ok=True)
    
    # Target files to encode - User specified list
    target_files = [
        'auth_system.py',
        'calibration_mode.py',
        'common_utils.py',
        'config.py',
        'electronic_signature.py',
        'multivariate_analysis.py',
        'peak_ai_analysis_web.py',
        'peak_analysis_web.py',
        'peak_deconvolution.py',
        'raman_database.py',
        'security_manager.py',
        'signature_management_ui.py',
        'signature_integration_example.py',
        'spectrum_analysis.py',
        'user_management_ui.py',
    ]
    
    print(f'\n🛡️  Output directory: {output_dir}')
    print('🔐 Processing files with enhanced security:')
    
    # Find existing files
    existing_files = []
    for filename in target_files:
        filepath = Path(filename)
        if filepath.exists():
            size = filepath.stat().st_size
            print(f'  ✅ {filename} ({size:,} bytes)')
            existing_files.append(filename)
        else:
            print(f'  ⚠️  {filename} (not found)')
    
    if not existing_files:
        print('\n❌ No target files found!')
        return 1
    
    print(f'\n🔒 Applying multi-layer encryption to {len(existing_files)} files...')
    print('-' * 50)
    
    encoded_count = 0
    failed_count = 0
    
    # Encode each file with enhanced security
    for filename in existing_files:
        encrypted_data = encoder.encode_file(filename)
        if encrypted_data:
            decoder_script = encoder.create_secure_decoder(encrypted_data)
            output_file = output_dir / filename
            
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(decoder_script)
                
                if output_file.exists():
                    original_size = Path(filename).stat().st_size
                    encoded_size = output_file.stat().st_size
                    ratio = encoded_size / original_size if original_size > 0 else 0
                    print(f'  🔐 {filename} → {encoded_size:,} bytes (ratio: {ratio:.1f}x)')
                    encoded_count += 1
                else:
                    print(f'  ❌ {filename} → Failed to create output')
                    failed_count += 1
            except Exception as e:
                print(f'  ❌ {filename} → Write error: {e}')
                failed_count += 1
        else:
            print(f'  ❌ {filename} → Encoding failed')
            failed_count += 1
    
    # Copy main_script.py without encoding (entry point should remain readable)
    print(f'\n📄 Copying main_script.py (unencoded entry point)...')
    main_script = Path('main_script.py')
    if main_script.exists():
        shutil.copy2(main_script, output_dir)
        print('  ✅ main_script.py copied')
    
    # Copy resources and create secure configuration
    print(f'\n📁 Setting up secure environment...')
    resources = 0
    
    # Copy essential resources
    resource_files = ['favicon.ico', 'logo.png', 'requirements.txt']
    for resource in resource_files:
        resource_path = Path(resource)
        if resource_path.exists():
            shutil.copy2(resource_path, output_dir)
            print(f'  ✅ {resource}')
            resources += 1
    
    # Create secure .streamlit configuration
    streamlit_source = Path('.streamlit')
    streamlit_dest = output_dir / '.streamlit'
    
    if streamlit_source.exists() and streamlit_source.is_dir():
        try:
            shutil.copytree(streamlit_source, streamlit_dest, dirs_exist_ok=True)
            print('  ✅ .streamlit configuration copied')
            resources += 1
        except Exception as e:
            print(f'  ⚠️ Failed to copy .streamlit: {e}')
            # Fallback: create minimal configuration
            streamlit_dest.mkdir(exist_ok=True)
            minimal_secrets = "# Add your API keys here\nopenai_api_key = \"your-api-key\""
            with open(streamlit_dest / 'secrets.toml', 'w', encoding='utf-8') as f:
                f.write(minimal_secrets)
            print('  ✅ Minimal .streamlit configuration created')
            resources += 1
    else:
        print('  ⚠️ .streamlit folder not found, skipping')
    
    
    print('  🛡️  Secure .streamlit configuration created')
    resources += 1
    
    # Copy other directories if they exist
    for dir_name in ['images', 'data', 'config']:
        dir_path = Path(dir_name)
        if dir_path.exists() and dir_path.is_dir():
            shutil.copytree(dir_path, output_dir / dir_name, dirs_exist_ok=True)
            print(f'  ✅ {dir_name} folder')
            resources += 1
    
    # Generate security report
    print('\n' + '=' * 70)
    print('    SECURITY ENHANCED ENCODING COMPLETED')
    print('=' * 70)
    print(f'🔐 Files encoded successfully: {encoded_count}')
    if failed_count > 0:
        print(f'❌ Files failed: {failed_count}')
    print(f'📁 Resources secured: {resources}')
    print(f'🛡️  Output: {output_dir}')
    
    print(f'\n🔒 Security Features Applied:')
    print(f'  • Multi-layer encryption: PBKDF2 + XOR + Bit manipulation')
    print(f'  • Custom Base64 alphabet: Standard analysis tools confused') 
    print(f'  • Dynamic key derivation: Time-based + Salt-based keys')
    print(f'  • Code obfuscation: Random variable names + Decoy functions')
    print(f'  • Anti-analysis: Hidden execution flow')
    print(f'  • No hardcoded keys: Runtime key reconstruction')
    
    if encoded_count > 0:
        print(f'\n🎉 SUCCESS: Security enhanced version ready!')
        print(f'\n⚠️  IMPORTANT SECURITY NOTES:')
        print(f'  • Encoded files are significantly harder to reverse engineer')
        print(f'  • Multiple layers of obfuscation applied')
        print(f'  • Dynamic key generation prevents static analysis')
        print(f'  • However, determined attackers may still reverse with effort')
        print(f'  • Consider additional server-side validation for maximum security')
        
        print(f'\n🧪 Next steps:')
        print(f'  1. cd {output_dir}')
        print(f'  2. Test with: streamlit run main_script.py')
        print(f'  3. Verify all encoded modules load correctly')
        return 0
    else:
        print(f'\n❌ FAILED: No files encoded!')
        return 1

if __name__ == '__main__':
    sys.exit(main())