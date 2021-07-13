.PHONY: install

install:
	cd "$(CURDIR)/gsl-2.6" && ./configure && $(MAKE) && $(MAKE) install prefix="$(CURDIR)/libs"
	cd .. && cp -r "gsl/lib/" "staircase/libs/gsl"
	
clean:
	cd "$(CURDIR)/gsl-2.6" && $(MAKE) clean
