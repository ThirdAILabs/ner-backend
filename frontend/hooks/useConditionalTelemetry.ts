import useTelemetry from './useTelemetry';
import { useEnterprise } from './useEnterprise';

export function useConditionalTelemetry() {
  const { isEnterprise, loading } = useEnterprise();
  const recordEvent = useTelemetry();

  return loading || isEnterprise ? () => {} : recordEvent;
}
