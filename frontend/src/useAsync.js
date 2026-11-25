import { useCallback, useEffect, useRef, useState } from 'react';

/**
 * Tracks one async call: { status: 'idle' | 'loading' | 'success' | 'error', data, error }.
 * Results that arrive after a newer call (or after unmount) are ignored.
 */
export function useAsync() {
  const [state, setState] = useState({ status: 'idle', data: null, error: null });
  const callId = useRef(0);
  const mounted = useRef(true);
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const run = useCallback(async (fn) => {
    const id = ++callId.current;
    setState((s) => ({ ...s, status: 'loading', error: null }));
    try {
      const data = await fn();
      if (mounted.current && id === callId.current)
        setState({ status: 'success', data, error: null });
      return data;
    } catch (error) {
      if (mounted.current && id === callId.current)
        setState({ status: 'error', data: null, error });
      return undefined;
    }
  }, []);

  return { ...state, run };
}
