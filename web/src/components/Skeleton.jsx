/** A grey shimmer block that holds a section's place while its data loads. */
export function Skeleton({ height = 16, width = "100%", style }) {
  return <div className="skeleton" aria-hidden="true" style={{ height, width, ...style }} />;
}

export function CardSkeleton({ lines = 3, height = 120 }) {
  return (
    <div className="card" role="status" aria-label="Loading">
      <Skeleton height={20} width="40%" style={{ marginBottom: 14 }} />
      {Array.from({ length: lines }, (_, i) => (
        <Skeleton key={i} height={height / lines} style={{ marginBottom: 8 }} />
      ))}
    </div>
  );
}
