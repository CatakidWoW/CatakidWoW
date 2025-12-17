export function invariant(cond: any, msg: string): asserts cond {
  if (!cond) throw new Error(msg);
}
