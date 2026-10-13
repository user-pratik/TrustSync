import React from 'react'
import clsx from 'clsx'

export interface SkeletonProps {
  className?: string
}

export const Skeleton: React.FC<SkeletonProps> = ({ className }) => {
  return (
    <div
      className={clsx(
        'animate-pulse rounded-md bg-white/[0.06]',
        className,
      )}
      aria-hidden="true"
    />
  )
}
