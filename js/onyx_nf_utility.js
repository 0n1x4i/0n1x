export function chainCallback(target, name, callback) {
  const previous = target?.[name];
  target[name] = function (...args) {
    const result = typeof previous === "function" ? previous.apply(this, args) : undefined;
    callback.apply(this, args);
    return result;
  };
}
