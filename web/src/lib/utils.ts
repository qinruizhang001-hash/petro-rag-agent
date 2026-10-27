import { type ClassValue, clsx } from "clsx";
import { twMerge } from "tailwind-merge";

/** shadcn 风格的类名合并。保留它，是为了将来 `npx shadcn add <comp>` 能直接接上。 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
