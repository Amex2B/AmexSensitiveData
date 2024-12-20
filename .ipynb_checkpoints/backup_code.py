# Phone number regex

def is_phone_number(phone):
    area_codes = [
            '205', '251', '256', '334', '659', '938',  # Alabama
            '907',  # Alaska
            '480', '520', '602', '623', '928',  # Arizona
            '327', '479', '501', '870',  # Arkansas
            '209', '213', '279', '310', '323', '341', '350', '369', '408', '415', '424', '442', '510', '530', '559', '562', '619', '626', '628', '650', '657', '661', '669', '707', '714', '747', '760', '805', '818', '820', '831', '840', '858', '909', '916', '925', '949', '951',  # California
            '303', '719', '720', '970', '983',  # Colorado
            '203', '475', '860', '959',  # Connecticut
            '302',  # Delaware
            '239', '305', '321', '324', '352', '386', '407', '448', '561', '645', '656', '689', '727', '728', '754', '772', '786', '813', '850', '863', '904', '941', '954',  # Florida
            '229', '404', '470', '478', '678', '706', '762', '770', '912', '943',  # Georgia
            '808',  # Hawaii
            '208', '986',  # Idaho
            '217', '224', '309', '312', '331', '447', '464', '618', '630', '708', '730', '773', '779', '815', '847', '861', '872',  # Illinois
            '219', '260', '317', '463', '574', '765', '812', '930',  # Indiana
            '319', '515', '563', '641', '712',  # Iowa
            '316', '620', '785', '913',  # Kansas
            '270', '364', '502', '606', '859',  # Kentucky
            '225', '318', '337', '504', '985',  # Louisiana
            '207',  # Maine
            '227', '240', '301', '410', '443', '667',  # Maryland
            '339', '351', '413', '508', '617', '774', '781', '857', '978',  # Massachusetts
            '231', '248', '269', '313', '517', '586', '616', '734', '810', '906', '947', '989',  # Michigan
            '218', '320', '507', '612', '651', '763', '952',  # Minnesota
            '228', '601', '662', '769',  # Mississippi
            '235', '314', '417', '557', '573', '636', '660', '816', '975',  # Missouri
            '406',  # Montana
            '308', '402', '531',  # Nebraska
            '702', '725', '775',  # Nevada
            '603',  # New Hampshire
            '201', '551', '609', '640', '732', '848', '856', '862', '908', '973',  # New Jersey
            '505', '575',  # New Mexico
            '212', '315', '329', '332', '347', '363', '516', '518', '585', '607', '624', '631', '646', '680', '716', '718', '838', '845', '914', '917', '929', '934',  # New York
            '252', '336', '472', '704', '743', '828', '910', '919', '980', '984',  # North Carolina
            '701',  # North Dakota
            '216', '220', '234', '283', '326', '330', '380', '419', '436', '440', '513', '567', '614', '740', '937',  # Ohio
            '405', '539', '572', '580', '918',  # Oklahoma
            '458', '503', '541', '971',  # Oregon
            '215', '223', '267', '272', '412', '445', '484', '570', '582', '610', '717', '724', '814', '835', '878',  # Pennsylvania
            '401',  # Rhode Island
            '803', '839', '843', '854', '864',  # South Carolina
            '605',  # South Dakota
            '423', '615', '629', '731', '865', '901', '931',  # Tennessee
            '210', '214', '254', '281', '325', '346', '361', '409', '430', '432', '469', '512', '682', '713', '726', '737', '806', '817', '830', '832', '903', '915', '936', '940', '945', '956', '972', '979',  # Texas
            '385', '435', '801',  # Utah
            '802',  # Vermont
            '276', '434', '540', '571', '686', '703', '757', '804', '826', '948',  # Virginia
            '206', '253', '360', '425', '509', '564',  # Washington
            '202', '771',  # Washington, DC
            '304', '681',  # West Virginia
            '262', '274', '353', '414', '534', '608', '715', '920',  # Wisconsin
            '307',  # Wyoming
            '684',  # American Samoa
            '671',  # Guam
            '670',  # Northern Mariana Islands
            '787', '939',  # Puerto Rico
            '340'  # Virgin Islands
        ]

    # Include area code in the criteria
    for pattern in patterns:
        if re.match(pattern, phone):
            # Extract the area code from the phone number
            match = re.search(r'\(?(\d{3})\)?[-\s.]?\d{3}[-\s.]?\d{4}', phone)
            if match:
                area_code = match.group(1)  # Extract the area code
                if area_code in area_codes:
                    return True  # The phone number is valid and has a valid area code
                else:
                    return False  # Invalid area code


# 10/3 previous code
# Regex pattern to extract potential phone numbers from text
phone_pattern = re.compile(r'[\d\s\-\.\(\)]{7,}')
# Function to extract and validate phone numbers
def extract_validate_phone(text):
    potential_phone = phone_pattern.findall(text)
    valid_phone = [num.strip() for num in potential_phone if is_phone_number(num.strip())]
    return valid_phone
# Apply the function to each row in the df
df['valid_phone'] = df['unmasked_text'].apply(extract_validate_phone)

# Display rows where valid phone numbers were found
valid_phone_df = df[df['valid_phone'].apply(len) > 0]

valid_phone_df

# Convert valid_phone_df into a csv file
valid_phone_df.to_csv('valid_phone_numbers.csv')
