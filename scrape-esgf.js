function test(elm) {
	href = elm.href;
	if (!href.includes('delete')) {
		href = href.slice('javascript:'.length, href.length);
		eval(href);
	}
}

document.querySelectorAll('.long > span > a').forEach((elm) => test(elm));
