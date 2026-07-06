const API_URL = "http://127.0.0.1:8000/api";

export async function fetchWithAuth(endpoint, options = {}) {
  const token = localStorage.getItem("accessToken");
  
  const headers = {
    ...options.headers,
  };

  // Only set Content-Type to application/json if we are not sending FormData
  if (!(options.body instanceof FormData)) {
    headers["Content-Type"] = "application/json";
  }

  if (token) {
    headers["Authorization"] = `Bearer ${token}`;
  }

  const response = await fetch(`${API_URL}${endpoint}`, {
    ...options,
    headers,
  });

  if (response.status === 401) {
    // Basic auto-logout on 401
    localStorage.removeItem("accessToken");
    localStorage.removeItem("refreshToken");
    window.location.href = "/";
    throw new Error("Session expired. Please log in again.");
  }

  if (!response.ok) {
    const errorData = await response.json().catch(() => ({}));
    throw new Error(errorData.error || `HTTP Error ${response.status}`);
  }

  return response.json();
}
