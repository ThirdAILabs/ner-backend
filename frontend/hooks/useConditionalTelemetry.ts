import useTelemetry from './useTelemetry';
import { useEnterprise } from './useEnterprise';

export function useConditionalTelemetry() {
  const { isEnterprise, enterpriseLoading } = useEnterprise();
  const recordEvent = useTelemetry();

  return enterpriseLoading || isEnterprise ? () => {} : recordEvent;
}
