interface ErrorMessageProps {
  message: string;
  onRetry?: () => void;
}

export function ErrorMessage({ message, onRetry }: ErrorMessageProps) {
  return (
    <div className="flex flex-col items-center gap-2 p-4 text-red-600">
      <p>{message}</p>
      {onRetry && (
        <button
          onClick={onRetry}
          className="px-3 py-1 text-sm bg-red-100 hover:bg-red-200 text-red-700 rounded"
        >
          重試
        </button>
      )}
    </div>
  );
}
