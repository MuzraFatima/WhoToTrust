import { useEffect, useState } from "react";

/**
 * Fetch with loading and error state. ``loading`` turns true in the same render in which the
 * inputs change (not one effect later), and an answer that arrives after the inputs changed again
 * is ignored. ``data`` keeps the previous answer while the next one loads, so lists do not flash.
 */
export function useAsync(load, deps) {
  const key = JSON.stringify(deps);
  const [state, setState] = useState({ key, data: null, error: null, loading: true });
  useEffect(() => {
    let live = true;
    setState((s) => ({ ...s, key, loading: true, error: null }));
    load().then(
      (data) => live && setState({ key, data, error: null, loading: false }),
      (error) => live && setState({ key, data: null, error, loading: false }),
    );
    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);
  const current = state.key === key;
  return {
    data: state.data,
    error: current ? state.error : null,
    loading: current ? state.loading : true,
  };
}

/** A value that follows `value` after it has stopped changing for `delay` ms. */
export function useDebounced(value, delay = 250) {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(id);
  }, [value, delay]);
  return debounced;
}

/** The current page from the URL hash (#/results), so reloads and back/forward work. */
export function useHashRoute(routes, fallback) {
  const read = () => {
    const name = window.location.hash.replace(/^#\/?/, "").split("?")[0];
    return routes.includes(name) ? name : fallback;
  };
  const [route, setRoute] = useState(read);
  useEffect(() => {
    const onChange = () => setRoute(read());
    window.addEventListener("hashchange", onChange);
    return () => window.removeEventListener("hashchange", onChange);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  return route;
}
