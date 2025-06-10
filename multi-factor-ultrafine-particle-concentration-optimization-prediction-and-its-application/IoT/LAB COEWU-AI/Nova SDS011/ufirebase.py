import urequests as requests
import json
import time
import gc

class UFirebase:
    def __init__(self, url, api_key, email, password):
        """Initialize Firebase with authentication credentials"""
        self.url = url.rstrip('/')
        self.api_key = api_key
        self.email = email
        self.password = password
        self.auth_token = None
        self.auth_expiry = 0
        gc.collect()  # Initial garbage collection
        
    def auth(self):
        """Authenticate with Firebase using email/password"""
        gc.collect()  # Free memory before authentication
        try:
            auth_url = "https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key=" + self.api_key
            auth_data = {
                "email": self.email,
                "password": self.password,
                "returnSecureToken": True
            }
            
            response = requests.post(
                auth_url,
                json=auth_data,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                auth_result = response.json()
                self.auth_token = auth_result.get('idToken')
                expires_in = int(auth_result.get('expiresIn', '3600'))
                self.auth_expiry = time.time() + expires_in
                
                # Clean up
                auth_result = None
                gc.collect()
                return True
            else:
                print("Authentication failed:", response.status_code)
                return False
                
        except Exception as e:
            print("Authentication error:", e)
            return False
        finally:
            if 'response' in locals():
                response.close()
                response = None
            gc.collect()
    
    def _check_token(self):
        """Check if token is expired and refresh if needed"""
        if not self.auth_token or time.time() > self.auth_expiry - 300:
            return self.auth()
        return True
    
    def put(self, path, data, chunk_size=256):
        """Send data to Firebase Realtime Database with chunked data handling"""
        gc.collect()  # Free memory before operation
        try:
            if not self._check_token():
                return False
            
            if not path:
                print("Invalid path")
                return False
            
            url = self.url + "/" + path + ".json?auth=" + self.auth_token
            
            # Convert data to JSON string in chunks if it's large
            try:
                json_data = json.dumps(data)
            except MemoryError:
                print("Data too large for available memory")
                return False
            
            response = requests.put(
                url,
                data=json_data,
                headers={'Content-Type': 'application/json'}
            )
            
            success = response.status_code == 200
            if not success:
                print("Firebase error:", response.status_code)
            
            return success
                
        except MemoryError:
            print("Memory error during PUT request")
            return False
        except Exception as e:
            print("Firebase put error:", e)
            return False
        finally:
            if 'response' in locals():
                response.close()
                response = None
            if 'json_data' in locals():
                json_data = None
            gc.collect()
    
    def get(self, path):
        """Get data from Firebase Realtime Database"""
        gc.collect()  # Free memory before operation
        try:
            if not self._check_token():
                return None
            
            url = self.url + "/" + path + ".json?auth=" + self.auth_token
            
            response = requests.get(url)
            
            if response.status_code == 200:
                # Process response in chunks if needed
                try:
                    result = response.json()
                    return result
                except MemoryError:
                    print("Response too large for available memory")
                    return None
            else:
                print("Firebase error:", response.status_code)
                return None
                
        except MemoryError:
            print("Memory error during GET request")
            return None
        except Exception as e:
            print("Firebase get error:", e)
            return None
        finally:
            if 'response' in locals():
                response.close()
                response = None
            if 'result' in locals():
                result = None
            gc.collect()

    def __del__(self):
        """Cleanup when object is deleted"""
        self.auth_token = None
        self.auth_expiry = 0
        gc.collect()