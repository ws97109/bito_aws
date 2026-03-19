export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

// Placeholder - real implementation would do: const res = await fetch(url); ...
export async function apiFetch<T>(url: string): Promise<T> {
  throw new ApiError(404, `Not found: ${url}`);
}
