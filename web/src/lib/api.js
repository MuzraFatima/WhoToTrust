// Small API client. GET answers are kept in memory for the page's lifetime (the data only changes
// when a run is rerun, which means reloading), and identical requests in flight share one fetch.
const cache = new Map();

export function clearApiCache() {
  cache.clear();
}

export async function getJSON(path, params = {}) {
  const qs = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === "" || value === false) continue;
    if (Array.isArray(value)) value.forEach((v) => qs.append(key, v));
    else qs.append(key, value);
  }
  const url = qs.size ? `${path}?${qs}` : path;
  if (!cache.has(url)) {
    const request = fetch(url, { headers: { Accept: "application/json" } })
      .then(async (res) => {
        const body = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(body.error || `${res.status} ${res.statusText}`);
        return body;
      })
      .catch((err) => {
        cache.delete(url); // do not keep failures
        throw err;
      });
    cache.set(url, request);
  }
  return cache.get(url);
}

export const imageUrl = (run, item) => `/api/image?${new URLSearchParams({ run, item })}`;
