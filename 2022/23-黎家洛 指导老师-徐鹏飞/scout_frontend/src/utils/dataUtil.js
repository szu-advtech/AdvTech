export const isNotNullORBlank = (...args)=> {
  for (var i = 0; i < args.length; i++) {
    var argument = args[i];
    if (argument == undefined || argument == null || argument === '' ) {
      return false;
    }
  }
  return true;
}
