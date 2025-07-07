import useTelemetry from './useTelemetry';
import { useEnterprise } from './useEnterprise';

export function useConditionalTelemetry() {
  const { isEnterprise, loading } = useEnterprise();
  const recordEvent = useTelemetry();

  console.log(`useConditionalTelemetry: isEnterprise=${isEnterprise}`);
  console.log(`useConditionalTelemetry: ${isEnterprise ? () => {} : recordEvent}`);

  return (loading || isEnterprise) ? () => {} : recordEvent;
}
