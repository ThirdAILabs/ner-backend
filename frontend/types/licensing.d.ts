interface License {
  LicenseInfo: {
    LicenseType: string;
    Expiry?: string;
    Usage: {
      MaxBytes: number;
      UsedBytes: number;
    };
  };
  LicenseError: string;
}
