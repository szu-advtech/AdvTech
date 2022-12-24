import argparse
from faker import Factory
import faker.providers
import itertools
import numpy.random as rand

fake = Factory.create()

providers = ['address', 'am_pm', 'binary', 'boolean', 'bothify', 'bs', 'building_number', 'catch_phrase', 'century', 'chrome', 'city', 'city_prefix', 'city_suffix', 'color_name', 'company', 'company_email', 'company_suffix', 'country', 'country_code', 'credit_card_expire', 'credit_card_full', 'credit_card_number', 'credit_card_provider', 'credit_card_security_code', 'currency_code', 'date', 'date_time', 'date_time_ad', 'date_time_between', 'date_time_between_dates', 'date_time_this_century', 'date_time_this_decade', 'date_time_this_month', 'date_time_this_year', 'day_of_month', 'day_of_week', 'domain_name', 'domain_word', 'ean', 'ean13', 'ean8', 'email', 'file_extension', 'file_name', 'firefox', 'first_name', 'first_name_female', 'first_name_male', 'free_email', 'free_email_domain', 'geo_coordinate', 'hex_color', 'image_url', 'internet_explorer', 'ipv4', 'ipv6', 'iso8601', 'job', 'language_code', 'last_name', 'last_name_female', 'last_name_male', 'latitude', 'lexify', 'linux_atform_token', 'linux_processor', 'locale', 'longitude', 'mac_address', 'mac_platform_token', 'mac_processor', 'md5', 'military_apo', 'military_dpo', 'military_ship', 'military_state', 'mime_type', 'month', 'month_name', 'name', 'name_female', 'name_male', 'null_boolean', 'numerify', 'opera', 'paragraph', 'paragraphs', 'password', 'phone_number', 'postalcode', 'postalcode_plus4', 'postcode', 'prefix', 'prefix_female', 'prefix_male', 'profile', 'provider', 'providers', 'pybool', 'pydecimal', 'pydict', 'pyfloat', 'pyint', 'pyiterable', 'pylist', 'pyset', 'pystr', 'pystruct', 'pytuple', 'random', 'random_digit', 'random_digit_not_null', 'random_digit_not_null_or_empty', 'random_digit_or_empty', 'random_element', 'random_int', 'random_letter', 'random_number', 'random_sample', 'random_sample_unique', 'randomize_nb_elements', 'rgb_color', 'rgb_color_list', 'rgb_css_color', 'safari', 'safe_color_name', 'safe_email', 'safe_hex_color', 'secondary_address', 'seed', 'sentence', 'sentences', 'set_formatter', 'sha1', 'sha256', 'simple_profile', 'slug', 'ssn', 'state', 'state_abbr', 'street_address', 'street_name', 'street_suffix', 'suffix', 'suffix_female', 'suffix_male', 'text', 'time', 'time_delta', 'timezone', 'tld', 'unix_time', 'uri', 'uri_extension', 'uri_page', 'uri_path', 'url', 'user_agent', 'user_name', 'uuid4', 'windows_platform_token', 'word', 'words', 'year', 'zipcode', 'zipcode_plus4']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generates a single-column text file with sample data obeying specified parameters")
    parser.add_argument('-f', '--filename', help='Relative path of the output file', required=True)
    parser.add_argument('-n', '--numrows', help='Number of rows to generate', required=True)
    parser.add_argument('-t', '--numtypes', help='Number of datatypes to mix', required=True)
    parser.add_argument('-st','--singletype', help="If n=1, specify this to specify a single type")
    parser.add_argument('-rl','--rowlength', help="Force row to truncate to certain number of chars")
    args = parser.parse_args()


    rows_per_type = int(args.numrows)/int(args.numtypes) + 1
    if int(args.numtypes) == 1 and args.singletype is not None:
        candidate_providers = [args.singletype]
    else:
        candidate_providers = rand.choice(providers, int(args.numtypes), False)
    lines = []
    for provider in candidate_providers:
        for i in range(rows_per_type):
            fake_entry = str(eval("fake." + provider + "()"))
            if args.rowlength is not None:
                if len(fake_entry) < int(args.rowlength):
                    print "Warning: row length > generated string"
                fake_entry = fake_entry[:int(args.rowlength)]
            lines.append(fake_entry)
    with open(args.filename, "w") as f:
        f.writelines('\n'.join(lines))
