I want to make helper functions that will simply operate on base cuda. Like you should be able to straight up swap out the op for the function.

So no random tiles that store data, you can just declare shared tiles or whatever, and use them to operate on the actual array. If you need to do other stuff, the array is just right there for you

```c++

// avoid this kinda stuff
shared_tile<...> s = al.allocate(...)
load(s, ...)

// do this
using s = shared_tile<...>;
__shared__ bf16 As[s.getSize()]
ldmatrix([s.getIdx]);

// idk if this is good design

```

Also just PTX helper functions are good.

### Things to do
- TK ops/group/memory/util has `load_async_wait` with a barrier id using bar.sync. Not sure how `bar.sync` works.