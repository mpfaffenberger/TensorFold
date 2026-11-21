//! Exec the installed sibling, never a binary from PATH or the checkpoint directory.
const std = @import("std");

pub fn run(init: std.process.Init, dir: []const u8, raw: []const []const u8, args: anytype) !u8 {
    for (raw) |arg| {
        if (!std.mem.startsWith(u8, arg, "--")) continue;
        const name = arg[0 .. std.mem.indexOfScalar(u8, arg, '=') orelse arg.len];
        for ([_][]const u8{ "--backend", "--host", "--port", "--context", "--max-tokens", "--name", "--vision" }) |allowed| {
            if (std.mem.eql(u8, name, allowed)) break;
        } else {
            std.debug.print("tensorfold serve: {s} is not supported by Inkling HIP\n", .{name});
            return 2;
        }
    }
    const a = init.arena.allocator();
    const exe = try std.process.executablePathAlloc(init.io, a);
    const helper = try std.fs.path.join(a, &.{ std.fs.path.dirname(exe).?, "tf-inkling-hip" });
    const port = try std.fmt.allocPrint(a, "{d}", .{args.port});
    const context = try std.fmt.allocPrint(a, "{d}", .{args.context orelse 4096});
    const tokens = try std.fmt.allocPrint(a, "{d}", .{args.max_tokens});
    const name = if (args.name.len != 0) args.name else args.model;
    const all = [_][]const u8{ helper, "serve", dir, "--host", args.host, "--port", port, "--context", context, "--max-tokens", tokens, "--name", name, "--vision" };
    const argv: []const []const u8 = if (args.vision) &all else all[0 .. all.len - 1];
    const err = std.process.replace(init.io, .{ .exe = .{ .path = .cwd() }, .argv = argv, .environ_map = init.environ_map });
    std.debug.print("tensorfold serve: cannot execute {s} ({s}); install the gfx1151 tf-inkling-hip build step\n", .{ helper, @errorName(err) });
    return 1;
}
